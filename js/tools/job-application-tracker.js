(() => {
  'use strict';
  const main = document.getElementById('main');
  if (!main) return;

  const configSource = document.body || main;
  const getDefaultRedirect = () => {
    try {
      const origin = (window.location && window.location.origin && window.location.origin !== 'null')
        ? String(window.location.origin)
        : '';
      const path = (window.location && window.location.pathname) ? String(window.location.pathname) : '';
      if (!origin || !path) return '';
      return `${origin}${path}`;
    } catch {
      return '';
    }
  };
  const normalizeRedirect = (value) => {
    const raw = (value || '').toString().trim();
    const fallback = getDefaultRedirect();
    if (!raw) return fallback;
    try {
      const url = new URL(raw, window.location.origin);
      if (url.origin !== window.location.origin) return fallback;
      return url.toString();
    } catch {
      return fallback;
    }
  };
  const config = {
    apiBase: (configSource.dataset.apiBase || '').trim(),
    cognitoDomain: (configSource.dataset.cognitoDomain || '').trim(),
    cognitoClientId: (configSource.dataset.cognitoClientId || '').trim(),
    cognitoRedirect: normalizeRedirect(configSource.dataset.cognitoRedirect),
    cognitoScopes: (configSource.dataset.cognitoScopes || 'openid email profile').trim(),
    maxAttachmentBytes: parseInt(configSource.dataset.maxAttachmentBytes || '10485760', 10) || 10485760,
    maxAttachmentCount: parseInt(configSource.dataset.maxAttachmentCount || '12', 10) || 12
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

	  const els = {
	    signIn: $('[data-jobtrack="sign-in"]'),
	    signOut: $('[data-jobtrack="sign-out"]'),
	    authModal: $('[data-jobtrack="auth-modal"]'),
	    jumpEntryButtons: $$('[data-jobtrack="jump-entry"]'),
	    jumpTabButtons: $$('[data-jobtrack-jump]'),
	    authStatus: $('[data-jobtrack="auth-status"]'),
	    apiStatus: $('[data-jobtrack="api-status"]'),
	    cognitoStatus: $('[data-jobtrack="cognito-status"]'),
    entryForm: $('[data-jobtrack="entry-form"]'),
    entryFormStatus: $('[data-jobtrack="entry-form-status"]'),
    entryDraftStatus: $('[data-jobtrack="entry-draft-status"]'),
    entryType: $('[data-jobtrack="entry-type"]'),
    entryTypeInputs: $$('[data-jobtrack="entry-type"] input[name="entryType"]'),
    companyInput: $('#jobtrack-company'),
    titleInput: $('#jobtrack-title'),
    jobUrlInput: $('#jobtrack-job-url'),
    jobUrlHelp: $('[data-jobtrack="job-url-help"]'),
    locationInput: $('#jobtrack-location'),
    sourceInput: $('#jobtrack-source'),
    batchInput: $('#jobtrack-batch'),
    appliedDateInput: $('#jobtrack-date'),
    appliedRequired: $('[data-jobtrack="applied-required"]'),
    postingDateInput: $('#jobtrack-posting-date'),
    postingUnknownInput: $('#jobtrack-posting-unknown'),
    captureDateInput: $('#jobtrack-capture-date'),
    captureLabel: $('[data-jobtrack="capture-label"]'),
    captureHelp: $('[data-jobtrack="capture-help"]'),
    statusInput: $('#jobtrack-status'),
    notesInput: $('#jobtrack-notes'),
    tagsInput: $('#jobtrack-tags'),
    followUpDateInput: $('#jobtrack-follow-up-date'),
    followUpNoteInput: $('#jobtrack-follow-up-note'),
    customFieldList: $('[data-jobtrack="custom-field-list"]'),
    customFieldAdd: $('[data-jobtrack="custom-field-add"]'),
    entrySubmit: $('[data-jobtrack="entry-submit"]'),
    entryApplicationFields: $$('[data-jobtrack-entry="application"]'),
    entryProspectFields: $$('[data-jobtrack-entry="prospect"]'),
    resumeInput: $('#jobtrack-resume'),
    coverInput: $('#jobtrack-cover'),
    resumePromptDownload: $('[data-jobtrack="resume-prompt-download"]'),
    coverPromptDownload: $('[data-jobtrack="cover-prompt-download"]'),
    importFile: $('#jobtrack-import-file'),
    importAttachments: $('#jobtrack-import-attachments'),
    importBatch: $('#jobtrack-import-batch'),
    importSubmit: $('[data-jobtrack="import-submit"]'),
    importTemplate: $('[data-jobtrack="import-template"]'),
    importStatus: $('[data-jobtrack="import-status"]'),
    importProgressWrap: $('[data-jobtrack="import-progress-wrap"]'),
    importProgressLabel: $('[data-jobtrack="import-progress-label"]'),
    importProgress: $('[data-jobtrack="import-progress"]'),
    prospectImportFile: $('#jobtrack-prospect-import-file'),
    prospectImportBatch: $('#jobtrack-prospect-import-batch'),
    prospectImportSubmit: $('[data-jobtrack="prospect-import-submit"]'),
    prospectImportTemplate: $('[data-jobtrack="prospect-import-template"]'),
    prospectPromptDownload: $('[data-jobtrack="prospect-prompt-download"]'),
    prospectImportStatus: $('[data-jobtrack="prospect-import-status"]'),
    prospectImportProgressWrap: $('[data-jobtrack="prospect-import-progress-wrap"]'),
    prospectImportProgressLabel: $('[data-jobtrack="prospect-import-progress-label"]'),
    prospectImportProgress: $('[data-jobtrack="prospect-import-progress"]'),
    prospectReviewList: $('[data-jobtrack="prospect-review-list"]'),
    prospectReviewStatus: $('[data-jobtrack="prospect-review-status"]'),
    prospectReviewRefresh: $('[data-jobtrack="prospect-review-refresh"]'),
    entryList: $('[data-jobtrack="entry-list"]'),
    entryListStatus: $('[data-jobtrack="entry-list-status"]'),
    entriesRefresh: $('[data-jobtrack="refresh-entries"]'),
    entryFilter: $('[data-jobtrack="entry-filter"]'),
    entryFilterQuery: $('[data-jobtrack="entry-filter-query"]'),
    entryFilterGroup: $('[data-jobtrack="entry-filter-group"]'),
    entryFilterType: $('[data-jobtrack="entry-filter-type"]'),
    entryFilterStatus: $('[data-jobtrack="entry-filter-status"]'),
    entryFilterSource: $('[data-jobtrack="entry-filter-source"]'),
    entryFilterBatch: $('[data-jobtrack="entry-filter-batch"]'),
    entryFilterLocation: $('[data-jobtrack="entry-filter-location"]'),
    entryFilterStart: $('[data-jobtrack="entry-filter-start"]'),
    entryFilterEnd: $('[data-jobtrack="entry-filter-end"]'),
    entryFilterTags: $('[data-jobtrack="entry-filter-tags"]'),
    entryFilterReset: $('[data-jobtrack="entry-filter-reset"]'),
    entrySortButtons: $$('[data-jobtrack-sort]'),
    entrySortSelect: $('[data-jobtrack="entry-sort-select"]'),
    entryViewWrap: $('[data-jobtrack="entry-view-wrap"]'),
    entryViewInputs: $$('[data-jobtrack="entry-view"] input[name="entryView"]'),
    entrySelectAll: $('[data-jobtrack="entry-select-all"]'),
    entryBulkDelete: $('[data-jobtrack="entry-bulk-delete"]'),
    entrySelectedCount: $('[data-jobtrack="entry-selected-count"]'),
    entryCount: $('[data-jobtrack="entry-count"]'),
    entrySummary: $('[data-jobtrack="entry-summary"]'),
    entryUndoBanner: $('[data-jobtrack="entry-undo"]'),
    entryUndoMessage: $('[data-jobtrack="entry-undo-message"]'),
    entryUndoAction: $('[data-jobtrack="entry-undo-action"]'),
    bulkStatusSelect: $('[data-jobtrack="bulk-status"]'),
    bulkStatusDate: $('[data-jobtrack="bulk-date"]'),
    bulkStatusApply: $('[data-jobtrack="bulk-status-apply"]'),
    savedViewSelect: $('[data-jobtrack="saved-view-select"]'),
    savedViewName: $('[data-jobtrack="saved-view-name"]'),
    savedViewSave: $('[data-jobtrack="saved-view-save"]'),
    savedViewDelete: $('[data-jobtrack="saved-view-delete"]'),
    savedViewStatus: $('[data-jobtrack="saved-view-status"]'),
    followupList: $('[data-jobtrack="followup-list"]'),
    followupStatus: $('[data-jobtrack="followup-status"]'),
    followupRefresh: $('[data-jobtrack="followup-refresh"]'),
    exportForm: $('[data-jobtrack="export-form"]'),
    exportStart: $('[data-jobtrack="export-start"]'),
    exportEnd: $('[data-jobtrack="export-end"]'),
    exportSubmit: $('[data-jobtrack="export-submit"]'),
    exportStatus: $('[data-jobtrack="export-status"]'),
    attachmentLimit: $('[data-jobtrack="attachment-limit"]'),
	    dashboard: $('[data-jobtrack="dashboard"]'),
	    dashboardStatus: $('[data-jobtrack="dashboard-status"]'),
	    dashboardTitle: $('[data-jobtrack="dashboard-title"]'),
	    dashboardSubtitle: $('[data-jobtrack="dashboard-subtitle"]'),
	    dashboardViewInputs: $$('[data-jobtrack="dashboard-view"] input[name="dashboardView"]'),
	    dashboardApplications: $('[data-jobtrack="dashboard-applications"]'),
	    dashboardProspects: $('[data-jobtrack="dashboard-prospects"]'),
	    dashboardProspectOpen: $('[data-jobtrack="dashboard-prospect-open"]'),
	    dashboardProspectStatus: $('[data-jobtrack="dashboard-prospect-status"]'),
	    filterStart: $('[data-jobtrack="filter-start"]'),
	    filterEnd: $('[data-jobtrack="filter-end"]'),
	    filterReset: $('[data-jobtrack="filter-reset"]'),
	    filterRefresh: $('[data-jobtrack="filter-refresh"]'),
    kpiTotal: $('[data-jobtrack="kpi-total"]'),
    kpiInterviews: $('[data-jobtrack="kpi-interviews"]'),
    kpiOffers: $('[data-jobtrack="kpi-offers"]'),
    kpiRejections: $('[data-jobtrack="kpi-rejections"]'),
    kpiFoundToApplied: $('[data-jobtrack="kpi-found-to-applied"]'),
    kpiFoundCount: $('[data-jobtrack="kpi-found-count"]'),
    kpiPostedToApplied: $('[data-jobtrack="kpi-posted-to-applied"]'),
    kpiPostedCount: $('[data-jobtrack="kpi-posted-count"]'),
    kpiResponseTime: $('[data-jobtrack="kpi-response-time"]'),
    kpiResponseTimeCount: $('[data-jobtrack="kpi-response-time-count"]'),
    kpiResponseRate: $('[data-jobtrack="kpi-response-rate"]'),
    kpiResponseCount: $('[data-jobtrack="kpi-response-count"]'),
    kpiTopSource: $('[data-jobtrack="kpi-top-source"]'),
    kpiTopSourceCount: $('[data-jobtrack="kpi-top-source-count"]'),
    kpiBestWeekday: $('[data-jobtrack="kpi-best-weekday"]'),
    kpiBestWeekdayCount: $('[data-jobtrack="kpi-best-weekday-count"]'),
    lineRange: $('[data-jobtrack="line-range"]'),
    lineTotal: $('[data-jobtrack="line-total"]'),
    statusTotal: $('[data-jobtrack="status-total"]'),
    calendarRange: $('[data-jobtrack="calendar-range"]'),
    lineOverlay: $('[data-jobtrack="line-overlay"]'),
    statusOverlay: $('[data-jobtrack="status-overlay"]'),
    calendarGrid: $('[data-jobtrack="calendar-grid"]'),
    calendarMonths: $('[data-jobtrack="calendar-months"]'),
    weekdayHeatmap: $('[data-jobtrack="weekday-heatmap"]'),
    mapContainer: $('[data-jobtrack="map"]'),
    mapPlaceholder: $('[data-jobtrack="map-placeholder"]'),
    mapTotal: $('[data-jobtrack="map-total"]'),
	    mapRemote: $('[data-jobtrack="map-remote"]'),
    funnelList: $('[data-jobtrack="funnel-list"]'),
    timeInStageList: $('[data-jobtrack="time-in-stage-list"]'),
	    kpiProspectsPending: $('[data-jobtrack="kpi-prospects-pending"]'),
	    kpiProspectsActive: $('[data-jobtrack="kpi-prospects-active"]'),
	    kpiProspectsInterested: $('[data-jobtrack="kpi-prospects-interested"]'),
	    kpiProspectsNeedsAction: $('[data-jobtrack="kpi-prospects-needs-action"]'),
	    prospectSummaryCount: $('[data-jobtrack="prospect-summary-count"]'),
	    prospectSummaryList: $('[data-jobtrack="prospect-summary-list"]'),
	    detailSubtitle: $('[data-jobtrack="detail-subtitle"]'),
	    detailBody: $('[data-jobtrack="detail-body"]'),
	    detailReset: $('[data-jobtrack="detail-reset"]'),
    detailModal: $('[data-jobtrack="detail-modal"]'),
    detailModalTitle: $('[data-jobtrack="detail-modal-title"]'),
    detailModalSubtitle: $('[data-jobtrack="detail-modal-subtitle"]'),
    detailModalBody: $('[data-jobtrack="detail-modal-body"]'),
    detailModalClose: $('[data-jobtrack="detail-modal-close"]'),
    detailModalStatus: $('[data-jobtrack="detail-modal-status"]')
  };

  const tabs = {
    buttons: $$('[data-jobtrack-tab]'),
    panels: $$('[data-jobtrack-panel]')
  };

		  const STORAGE_KEY = 'jobTrackerAuth';
		  const SHARED_STORAGE_KEY = 'toolsAuth';
		  const STATE_KEY = 'jobTrackerAuthState';
		  const VERIFIER_KEY = 'jobTrackerCodeVerifier';
		  const RETURN_TAB_KEY = 'jobTrackerReturnTab';
		  const ENTRY_VIEW_KEY = 'jobTrackerEntryView';
		  const DASHBOARD_VIEW_KEY = 'jobTrackerDashboardView';
		  const ENTRY_DRAFT_KEY = 'jobTrackerEntryDraft';
  const CSV_TEMPLATE = 'company,title,jobUrl,location,source,postingDate,appliedDate,status,batch,notes,tags,followUpDate,followUpNote,customFields,attachments\nAcme Corp,Data Analyst,https://acme.com/jobs/123,Remote,LinkedIn,2025-01-10,2025-01-15,Applied,Spring outreach 2025,Reached out to recruiter,referral;remote,2025-01-20,Nudge recruiter after screening,"{\"salary\":\"120k\",\"priority\":\"High\"}",Acme-Resume.pdf;Acme-Cover.pdf';
  const PROSPECT_CSV_TEMPLATE = 'company,title,jobUrl,location,source,postingDate,captureDate,status,batch,notes,tags,followUpDate,followUpNote,customFields\nAcme Corp,Data Analyst,https://acme.com/jobs/123,Remote,LinkedIn,2025-01-10,2025-01-12,Active,Remote data roles · March,Follow up next week,remote;priority,2025-01-18,Review again Friday,"{\"priority\":\"High\"}"';
  const PROSPECT_PROMPT_TEMPLATE = [
    'Prompt:',
    'Using reputable sources and live job data, identify data analyst, data scientist, machine learning engineer, research scientist, analytics engineer, or closely related roles that align with my preferences below. Treat the strict criteria as non-negotiable unless otherwise marked.',
    '',
    'MY JOB SEARCH CRITERIA',
    '',
    '1) Role Types (strict)',
    '- Data Analyst',
    '- Data Scientist',
    '- Machine Learning Engineer',
    '- Research Scientist (AI/ML)',
    '- Applied Scientist',
    '- Analytics Engineer',
    '- Quantitative/Data-heavy Analyst',
    '- Plus any roles historically shown in our past chats that match my skillset and interests (deep analytics, ML modeling, SQL, Python, cloud, experimentation, etc.)',
    '',
    '2) Posting Freshness (strict)',
    '- Only include roles posted within the last 7 days.',
    '',
    '3) Source Requirements (strict)',
    '- Direct from company website or ATS (Workday, Greenhouse, Lever, Taleo, etc.)',
    '- No job boards such as LinkedIn, Indeed, ZipRecruiter, Glassdoor, or aggregated feeds.',
    '',
    '4) Location Preferences (flexible but prioritized)',
    '1. Remote (top preference)',
    '2. Grand Junction, CO',
    '3. Dallas-Fort Worth (DFW)',
    '4. Louisiana',
    '5. Other U.S. locations with friendly home-schooling laws',
    '6. Any U.S. location if company stability + work-life balance are strong',
    '',
    '5) Salary Target (flexible)',
    '- Ideal: >= $90,000',
    '- Acceptable: Slight step down if work-life balance + job stability are strong',
    '- If salary is unlisted, provide a researched or inferred range based on similar roles at that company.',
    '',
    '6) Company Stability + Layoff Filter (strict)',
    '- Exclude companies with recent layoffs (2024-2025), recurring layoffs, hiring freezes, restructurings, or major uncertainty.',
    '- Prioritize stable headcounts, positive or neutral news, established enterprises, government, education, healthcare, utilities, and highly stable sectors. Include late-stage, well-capitalized companies if private.',
    '',
    '7) Work-Life Balance + Reviews (strict priority)',
    '- Evaluate work-life balance reputation, recent reviews (2024-2025), sentiment toward management, burnout, and expectations.',
    '- Exclude companies with consistent burnout complaints, toxic culture, poor management ratings, or extreme overtime expectations.',
    '',
    'OUTPUT FORMAT (STRICT, CSV FOR JOB TRACKER PROSPECTS IMPORT)',
    '- Return only CSV (no markdown).',
    '- Header must be: company,title,jobUrl,location,source,postingDate,captureDate,status,batch,notes,tags,followUpDate,followUpNote,customFields',
    '- One role per row. Quote any field that contains commas.',
    '- jobUrl must be a direct company/ATS link.',
    '- source should be the site name (Company site, Workday, Greenhouse, Lever, Taleo, etc).',
    '- postingDate must be within the last 7 days and in YYYY-MM-DD. If unknown, leave blank.',
    '- captureDate should be today in YYYY-MM-DD.',
    '- status must be Active.',
    '- location should include the city/state or region plus classification in parentheses (Remote/Hybrid/On-site).',
    '- batch should be a shared label for this search (e.g., "Remote data roles · March"), or leave blank if you will add it in the import form.',
    '- notes should be one short line that includes: why the role fits, estimated salary range, stability signal, and work-life balance snapshot.',
    '- tags should be semicolon-separated labels like "remote;priority".',
    '- followUpDate and followUpNote are optional; leave blank if not needed.',
    '- customFields should be JSON (e.g., {"priority":"High","salary":"90k"}) or blank.',
    '- End with a weekly hiring trend summary by appending "Weekly trend: ..." to the notes field of the final job row.'
  ].join('\n');
  const APPLICATION_STATUSES = ['Applied', 'Screening', 'Interview', 'Offer', 'Rejected', 'Withdrawn'];
  const PROSPECT_STATUSES = ['Active', 'Interested', 'Rejected', 'Inactive'];
  const STATUS_GROUPS = {
    active: new Set(['applied', 'screening', 'interview', 'offer', 'active', 'interested']),
    archived: new Set(['withdrawn', 'inactive']),
    rejected: new Set(['rejected'])
  };
  const FOLLOW_UP_DAYS = {
    applied: 7,
    screening: 5,
    interview: 3,
    active: 5,
    interested: 3
  };
  const FOLLOWUP_RANGE_DAYS = 14;
	  const DETAIL_DEFAULT_SUBTITLE = 'Click a chart element to inspect activity.';
	  const DETAIL_DEFAULT_BODY = 'Select a state, week, weekday, day, or status to see details here.';
	  const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
	  const REMOTE_HINTS = ['remote', 'work from home', 'wfh', 'virtual'];
	  const HYBRID_HINTS = ['hybrid'];
	  const ONSITE_HINTS = ['on-site', 'onsite', 'on site', 'in office', 'in-office', 'in person', 'in-person'];
  const US_STATES = [
    { code: 'AL', name: 'Alabama' },
    { code: 'AK', name: 'Alaska' },
    { code: 'AZ', name: 'Arizona' },
    { code: 'AR', name: 'Arkansas' },
    { code: 'CA', name: 'California' },
    { code: 'CO', name: 'Colorado' },
    { code: 'CT', name: 'Connecticut' },
    { code: 'DE', name: 'Delaware' },
    { code: 'FL', name: 'Florida' },
    { code: 'GA', name: 'Georgia' },
    { code: 'HI', name: 'Hawaii' },
    { code: 'ID', name: 'Idaho' },
    { code: 'IL', name: 'Illinois' },
    { code: 'IN', name: 'Indiana' },
    { code: 'IA', name: 'Iowa' },
    { code: 'KS', name: 'Kansas' },
    { code: 'KY', name: 'Kentucky' },
    { code: 'LA', name: 'Louisiana' },
    { code: 'ME', name: 'Maine' },
    { code: 'MD', name: 'Maryland' },
    { code: 'MA', name: 'Massachusetts' },
    { code: 'MI', name: 'Michigan' },
    { code: 'MN', name: 'Minnesota' },
    { code: 'MS', name: 'Mississippi' },
    { code: 'MO', name: 'Missouri' },
    { code: 'MT', name: 'Montana' },
    { code: 'NE', name: 'Nebraska' },
    { code: 'NV', name: 'Nevada' },
    { code: 'NH', name: 'New Hampshire' },
    { code: 'NJ', name: 'New Jersey' },
    { code: 'NM', name: 'New Mexico' },
    { code: 'NY', name: 'New York' },
    { code: 'NC', name: 'North Carolina' },
    { code: 'ND', name: 'North Dakota' },
    { code: 'OH', name: 'Ohio' },
    { code: 'OK', name: 'Oklahoma' },
    { code: 'OR', name: 'Oregon' },
    { code: 'PA', name: 'Pennsylvania' },
    { code: 'RI', name: 'Rhode Island' },
    { code: 'SC', name: 'South Carolina' },
    { code: 'SD', name: 'South Dakota' },
    { code: 'TN', name: 'Tennessee' },
    { code: 'TX', name: 'Texas' },
    { code: 'UT', name: 'Utah' },
    { code: 'VT', name: 'Vermont' },
    { code: 'VA', name: 'Virginia' },
    { code: 'WA', name: 'Washington' },
    { code: 'WV', name: 'West Virginia' },
    { code: 'WI', name: 'Wisconsin' },
    { code: 'WY', name: 'Wyoming' },
    { code: 'DC', name: 'District Of Columbia' }
  ];
  const STATE_CODE_SET = new Set(US_STATES.map(state => state.code.toLowerCase()));
  const STATE_NAME_LOOKUP = new Map(US_STATES.map(state => [state.code, state.name]));
  const STATE_NAME_BY_LOWER = new Map(US_STATES.map(state => [state.name.toLowerCase(), state.code]));

	  const state = {
	    auth: null,
	    lineChart: null,
	    statusChart: null,
	    range: null,
    editingEntryId: null,
    editingEntry: null,
	    entryType: 'application',
	    entries: [],
	    entriesLoaded: false,
	    followups: [],
	    dashboardEntries: [],
	    entryItems: new Map(),
	    entrySort: { key: 'date', direction: 'desc' },
	    entryView: 'table',
	    dashboardView: 'applications',
	    savedViews: [],
	    activeViewId: '',
	    entryFilters: {
      query: '',
      type: 'all',
      status: 'all',
      location: 'all',
      start: '',
      end: ''
    },
    selectedEntryIds: new Set(),
    visibleEntryIds: [],
    entryDraftTimer: null,
    entryUndoTimer: null,
    lastDeletedEntry: null,
    mapLoaded: false,
    mapSvg: null,
    mapDetails: null,
    calendarDetails: null,
    statusDetails: null,
    weekDetails: null,
    weekdayDetails: null,
    weeklySeries: [],
    detailModalEntryId: null,
    detailModalPrevFocus: null,
    mapTooltipBound: false,
    calendarTooltipBound: false,
	    weekdayTooltipBound: false,
	    isResettingEntry: false
	  };

	  let authExpiryTimer = null;
	  let authModalPrevFocus = null;
	  let authModalTrapBound = false;
	  let authUiMessage = '';
	  let authUiTone = '';
	  const authModalBackgroundEls = [
	    $('.skip-link'),
	    $('#combined-header-nav'),
	    $('.jobtrack-hero'),
	    main,
	    $('footer')
	  ].filter(Boolean);

	  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const createTooltip = () => {
    const el = document.createElement('div');
    el.className = 'jobtrack-tooltip';
    el.dataset.state = 'hidden';
    el.setAttribute('role', 'tooltip');
    el.setAttribute('aria-hidden', 'true');
    document.body.appendChild(el);

    const position = (x, y) => {
      const offset = 14;
      const rect = el.getBoundingClientRect();
      let left = x + offset;
      let top = y + offset;
      if (left + rect.width > window.innerWidth - 12) {
        left = x - rect.width - offset;
      }
      if (top + rect.height > window.innerHeight - 12) {
        top = y - rect.height - offset;
      }
      left = Math.max(8, Math.min(left, window.innerWidth - rect.width - 8));
      top = Math.max(8, Math.min(top, window.innerHeight - rect.height - 8));
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
    };

    const show = (text, x, y) => {
      if (!text) return;
      if (el.textContent !== text) el.textContent = text;
      if (el.dataset.state !== 'visible') {
        el.dataset.state = 'visible';
        el.setAttribute('aria-hidden', 'false');
      }
      position(x, y);
    };

    const hide = () => {
      if (el.dataset.state !== 'visible') return;
      el.dataset.state = 'hidden';
      el.setAttribute('aria-hidden', 'true');
    };

    window.addEventListener('scroll', hide, { passive: true });
    window.addEventListener('resize', hide, { passive: true });

    return { show, hide };
  };

  const tooltip = createTooltip();

  const runWithConcurrency = async (items, limit, task) => {
    const results = [];
    let index = 0;
    const runWorker = async () => {
      while (index < items.length) {
        const current = items[index];
        index += 1;
        try {
          const value = await task(current);
          results.push({ ok: true, value });
        } catch (err) {
          results.push({ ok: false, error: err });
        }
      }
    };
    const workerCount = Math.max(1, Math.min(limit, items.length));
    const workers = Array.from({ length: workerCount }, () => runWorker());
    await Promise.all(workers);
    return results;
  };

  const formatDateInput = (date) => date.toISOString().slice(0, 10);
  const parseDateInput = (value) => {
    if (!value) return null;
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };
  const parseIsoDate = (value) => {
    if (!value) return null;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };
  const parseHistoryDate = (value) => parseDateInput(value) || parseIsoDate(value);
  const formatDateLabel = (value) => {
    const parsed = parseDateInput(value) || parseIsoDate(value);
    if (!parsed) return '—';
    return parsed.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };
  const formatDays = (value) => {
    if (!Number.isFinite(value)) return '--';
    return `${value.toFixed(1)} days`;
  };
  const formatPercent = (value) => {
    if (!Number.isFinite(value)) return '--';
    return `${Math.round(value)}%`;
  };

  const toPromptValue = (value) => (value || '').toString().trim();
  const formatPromptValue = (value, fallback = 'Not provided') => {
    const trimmed = toPromptValue(value);
    return trimmed ? trimmed : fallback;
  };
  const mergePromptContext = (base, overrides) => {
    const merged = { ...base };
    Object.keys(overrides).forEach((key) => {
      if (overrides[key]) merged[key] = overrides[key];
    });
    return merged;
  };
  const buildPromptContextFromEntry = (entry = {}) => {
    const jobUrl = normalizeUrl(entry.jobUrl || entry.url || '');
    return {
      company: toPromptValue(entry.company),
      title: toPromptValue(entry.title),
      jobUrl,
      location: toPromptValue(entry.location),
      source: toPromptValue(entry.source),
      postingDate: parseDateInput(entry.postingDate) ? entry.postingDate : '',
      appliedDate: parseDateInput(entry.appliedDate) ? entry.appliedDate : '',
      captureDate: parseDateInput(entry.captureDate) ? entry.captureDate : '',
      status: toPromptValue(entry.status),
      batch: toPromptValue(entry.batch),
      notes: toPromptValue(entry.notes)
    };
  };
  const buildPromptContextFromForm = () => {
    const base = state.editingEntry ? buildPromptContextFromEntry(state.editingEntry) : {};
    const formValues = {
      company: toPromptValue(els.companyInput?.value),
      title: toPromptValue(els.titleInput?.value),
      jobUrl: normalizeUrl(toPromptValue(els.jobUrlInput?.value)),
      location: toPromptValue(els.locationInput?.value),
      source: toPromptValue(els.sourceInput?.value),
      postingDate: parseDateInput(els.postingDateInput?.value) ? els.postingDateInput.value : '',
      appliedDate: parseDateInput(els.appliedDateInput?.value) ? els.appliedDateInput.value : '',
      captureDate: parseDateInput(els.captureDateInput?.value) ? els.captureDateInput.value : '',
      status: toPromptValue(els.statusInput?.value),
      batch: toPromptValue(els.batchInput?.value),
      notes: toPromptValue(els.notesInput?.value)
    };
    return mergePromptContext(base, formValues);
  };
  const buildPromptHeader = (context = {}) => {
    const lines = [
      'Job details (from tracker):',
      `Company: ${formatPromptValue(context.company)}`,
      `Role: ${formatPromptValue(context.title)}`,
      `Job description URL: ${formatPromptValue(context.jobUrl)}`,
      `Location: ${formatPromptValue(context.location)}`,
      `Source: ${formatPromptValue(context.source)}`,
      `Posting date: ${formatPromptValue(context.postingDate)}`,
      `Applied date: ${formatPromptValue(context.appliedDate)}`,
      `Found date: ${formatPromptValue(context.captureDate)}`,
      `Status: ${formatPromptValue(context.status)}`,
      `Batch: ${formatPromptValue(context.batch)}`,
      `Notes: ${formatPromptValue(context.notes)}`
    ];
    return lines.join('\n');
  };
  const buildNaturalLanguagePrompt = (topicOrRequest) => {
    const topic = toPromptValue(topicOrRequest) || '[INSERT YOUR TOPIC OR REQUEST HERE]';
    return [
      'Act like a professional content writer and communication strategist. Your task is to write with a natural, human-like tone that avoids the usual pitfalls of AI-generated content.',
      '',
      'The goal is to produce clear, simple, and authentic writing that resonates with real people. Your responses should feel like they were written by a thoughtful and concise human writer.',
      '',
      'You are writing the following:',
      topic,
      '',
      'Follow these detailed step-by-step guidelines:',
      '',
      'Step 1: Use plain and simple language. Avoid long or complex sentences. Opt for short, clear statements.',
      '',
      '- Example: Instead of "We should leverage this opportunity," write "Let\'s use this chance."',
      '',
      'Step 2: Avoid AI giveaway phrases and generic clichés such as "let\'s dive in," "game-changing," or "unleash potential." Replace them with straightforward language.',
      '',
      '- Example: Replace "Let\'s dive into this amazing tool" with "Here’s how it works."',
      '',
      'Step 3: Be direct and concise. Eliminate filler words and unnecessary phrases. Focus on getting to the point.',
      '',
      '- Example: Say "We should meet tomorrow," instead of "I think it would be best if we could possibly try to meet."',
      '',
      'Step 4: Maintain a natural tone. Write like you speak. It’s okay to start sentences with “and” or “but.” Make it feel conversational, not robotic.',
      '',
      '- Example: “And that’s why it matters.”',
      '',
      'Step 5: Avoid marketing buzzwords, hype, and overpromises. Use neutral, honest descriptions.',
      '',
      '- Avoid: "This revolutionary app will change your life."',
      '- Use instead: "This app can help you stay organized."',
      '',
      'Step 6: Keep it real. Be honest. Don’t try to fake friendliness or exaggerate.',
      '',
      '- Example: “I don’t think that’s the best idea.”',
      '',
      'Step 7: Simplify grammar. Don’t worry about perfect grammar if it disrupts natural flow. Casual expressions are okay.',
      '',
      '- Example: “i guess we can try that.”',
      '',
      'Step 8: Remove fluff. Avoid using unnecessary adjectives or adverbs. Stick to the facts or your core message.',
      '',
      '- Example: Say “We finished the task,” not “We quickly and efficiently completed the important task.”',
      '',
      'Step 9: Focus on clarity. Your message should be easy to read and understand without ambiguity.',
      '',
      '- Example: “Please send the file by Monday.”',
      '',
      'Follow this structure rigorously. Your final writing should feel honest, grounded, and like it was written by a clear-thinking, real person.',
      '',
      'Take a deep breath and work on this step-by-step.'
    ];
  };
  const buildResumePrompt = (context = {}) => {
    const jobUrl = formatPromptValue(context.jobUrl);
    return [
      buildPromptHeader(context),
      '',
      'Prompt for Optimizing My Resume Using a Job Description and Personal Website',
      '',
      'Act as an expert resume writer and career coach for data-focused roles. I need you to optimize my existing resume to align with a specific job description. Use the following inputs:',
      '',
      '1. Resume content (attached as a Word document). It describes my roles as a Business Analyst, AI Data Quality Analyst, and Asset Protection Data Analyst, including quantified achievements like reducing reporting time by 99%, lifting site traffic by 750%, saving 200+ hours per year, and increasing theft prevention by 180%.',
      `2. Job description URL: ${jobUrl}.`,
      '3. Portfolio website: danielshort.me. This site lists my projects and highlights key metrics (e.g., 94% RL solver accuracy, 200+ hours saved through automation, 750% traffic lift). For this resume, focus on three projects - Chatbot (LoRA + RAG), Shape Classifier Demo, and Sheet Music Restoration - and ignore the others. It also includes sections on certifications, degrees and demonstrated strengths (Python, SQL & ETL, Tableau dashboards, data wrangling, reinforcement learning, regression analysis).',
      '',
      ...buildNaturalLanguagePrompt('Rewrite my resume in plain text.'),
      '',
      'Instructions:',
      '- Review the job description to identify core responsibilities, must-have skills, and preferred qualifications.',
      '- Analyze my resume and note any gaps or misalignments relative to the job description. Use the additional details from danielshort.me (project descriptions, metrics, certifications, strengths) to enrich the resume where appropriate.',
      '- Rewrite the professional summary to highlight my most relevant experience, drawing on the website\'s metrics (e.g., 750% traffic lift, 200+ hours saved, 94% model accuracy) and emphasizing skills that match the job description.',
      '- Refine each experience bullet point so it begins with a strong action verb, incorporates tools/technologies used (e.g., PyTorch, SQL, RAG) and quantifies results. Align the emphasis with the job description - focus on automation and data-pipeline achievements if the role stresses MLOps; highlight generative-AI work if LLMs or RAG are mentioned.',
      '- Update the skills section to group technical skills logically (Programming & ML frameworks, Data Engineering & SQL, Visualization, Soft skills) and prioritize those required by the job description. Include certifications and strengths from the website\'s "Certifications & Degrees" and "Demonstrated Strengths" sections.',
      '- Expand the projects section by summarizing only three projects - Chatbot (LoRA + RAG), Shape Classifier Demo, and Sheet Music Restoration. For each, describe the problem, approach, and outcome using the descriptions from the website. Do not include other projects.',
      '- Maintain ATS-friendly formatting - use standard section headings, avoid tables or images, and aim for a concise one-page length.',
      '',
      'Return a revised resume in plain text with updated summary, experience bullets, skills, projects, education and certifications. Note any assumptions or additional information you needed to make alignment recommendations.'
    ].join('\n');
  };
  const buildCoverLetterPrompt = (context = {}) => {
    const role = toPromptValue(context.title) || 'the role';
    const company = toPromptValue(context.company) || 'the company';
    return [
      buildPromptHeader(context),
      '',
      `You are a professional cover letter writer. I am applying for the position of ${role} at ${company}. I have attached my resume as a file. Please research ${company} from reliable sources (e.g., official website, recent news) to understand its mission, values, and current initiatives, and use that context to craft a personalized, one-page cover letter.`,
      '',
      'Attached Resume:',
      '(Use the contents of the attached resume file provided.)',
      '',
      ...buildNaturalLanguagePrompt('A tailored, one-page cover letter.'),
      '',
      'Instructions:',
      '1. Refer to the attached resume file to identify my key roles, achievements, and skills.',
      `2. Look up ${company} to find its mission statement, values, products or services, and any noteworthy recent projects or goals.`,
      '3. Write a tailored cover letter that:',
      '- Starts with a personal greeting addressed to the hiring manager or relevant department (use "Dear Hiring Team" if no name is known).',
      '- In the opening paragraph, states the role I am applying for and expresses genuine enthusiasm for the company, referencing specific company values, goals, or recent initiatives discovered during your research.',
      '- In the body, highlights 1-2 achievements from my resume that illustrate how my skills match the key requirements of the job description, quantifying results where possible and explaining context, actions, and outcomes.',
      '- Discusses how the company\'s mission resonates with me and how my experience aligns with its goals.',
      '- Maintains a professional yet conversational tone; avoids cliches and overly formal language; keeps the letter under three or four concise paragraphs.',
      '- Concludes by summarizing why I am a strong fit, expressing appreciation for the opportunity, and inviting further discussion. Use a professional sign-off (e.g., "Sincerely").',
      '',
      `Ensure the cover letter sounds authentic and reflects my unique voice, demonstrating that you have genuinely researched ${company}. Remove any generic or robotic phrasing. If any details in the resume are unclear, ask clarifying questions. Please write the cover letter now.`
    ].join('\n');
  };
  const buildPromptFilename = (context = {}, type = 'prompt') => {
    const parts = [context.company, context.title, type]
      .map(value => toPromptValue(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''))
      .filter(Boolean);
    const base = parts.join('-');
    return base ? `${base}.txt` : `${type}.txt`;
  };
  const downloadPromptFile = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };
  const downloadEntryPrompt = (context, type) => {
    const content = type === 'cover'
      ? buildCoverLetterPrompt(context)
      : buildResumePrompt(context);
    const filename = buildPromptFilename(context, type === 'cover' ? 'cover-letter-prompt' : 'resume-prompt');
    downloadPromptFile(content, filename);
  };

  const deriveStatusDate = (entry) => {
    if (!entry) return '';
    const direct = (entry.statusDate || '').toString().trim();
    if (direct && parseDateInput(direct)) return direct;
    const history = Array.isArray(entry.statusHistory) ? entry.statusHistory : [];
    if (!history.length) return '';
    const currentStatus = (entry.status || '').toString().trim().toLowerCase();
    let latestForStatus = null;
    let latestAny = null;
    history.forEach((item) => {
      const date = parseHistoryDate(item?.date);
      if (!date) return;
      if (!latestAny || date > latestAny) latestAny = date;
      const status = (item?.status || '').toString().trim().toLowerCase();
      if (currentStatus && status === currentStatus) {
        if (!latestForStatus || date > latestForStatus) latestForStatus = date;
      }
    });
    const target = latestForStatus || latestAny;
    return target ? formatDateInput(target) : '';
  };

  const incrementCount = (map, key) => {
    if (!key) return;
    map.set(key, (map.get(key) || 0) + 1);
  };

	  const formatCountList = (map, limit = 4) => {
	    const items = Array.from(map.entries())
	      .sort((a, b) => b[1] - a[1])
	      .slice(0, limit)
	      .map(([label, count]) => `${label} (${count})`);
	    return items.length ? items.join(', ') : '--';
	  };

	  const extractStateCodeFromText = (location = '') => {
	    const raw = (location || '').toString();
	    if (!raw.trim()) return null;
	    const upper = raw.toUpperCase();
	    let bestIndex = Number.POSITIVE_INFINITY;
	    let bestCode = null;
	    const codePattern = /\b([A-Z]{2})\b/g;
	    let match = null;
	    while ((match = codePattern.exec(upper))) {
	      const code = match[1];
	      if (!STATE_CODE_SET.has(code.toLowerCase())) continue;
	      if (match.index < bestIndex) {
	        bestIndex = match.index;
	        bestCode = code;
	      }
	    }

	    const lower = raw.toLowerCase();
	    for (const [nameLower, code] of STATE_NAME_BY_LOWER.entries()) {
	      const index = lower.indexOf(nameLower);
	      if (index === -1) continue;
	      const before = index === 0 ? '' : lower[index - 1];
	      const afterIndex = index + nameLower.length;
	      const after = afterIndex >= lower.length ? '' : lower[afterIndex];
	      const beforeOk = !before || /[^a-z]/.test(before);
	      const afterOk = !after || /[^a-z]/.test(after);
	      if (!beforeOk || !afterOk) continue;
	      if (index < bestIndex) {
	        bestIndex = index;
	        bestCode = code;
	      }
	    }
	    return bestCode;
	  };

	  const cleanLocationLabel = (location = '') => {
	    const raw = (location || '').toString().trim();
	    if (!raw) return '';
	    const withoutTags = raw
	      .replace(/\(\s*(?:remote|wfh|work from home|virtual)\s*\)/gi, '')
	      .replace(/\(\s*(?:hybrid)\s*\)/gi, '')
	      .replace(/\(\s*(?:on[- ]?site|onsite|on site|in[- ]office|in office|in[- ]person|in person)\s*\)/gi, '')
	      .replace(/[-–—]\s*(?:remote|wfh)\b/gi, '')
	      .replace(/[-–—]\s*hybrid\b/gi, '')
	      .replace(/[-–—]\s*(?:on[- ]?site|onsite|on site|in[- ]office|in office|in[- ]person|in person)\b/gi, '');
	    const normalized = withoutTags
	      .replace(/\s*\/\s*/g, ' / ')
	      .replace(/\s*,\s*/g, ', ')
	      .replace(/\(\s*\)/g, '')
	      .replace(/\(\s+/g, '(')
	      .replace(/\s+\)/g, ')')
	      .replace(/\s{2,}/g, ' ')
	      .replace(/(?:\s*\/\s*)+$/g, '')
	      .trim();
	    return normalized;
	  };

	  const parseLocation = (location = '') => {
	    const raw = (location || '').toString().trim();
	    if (!raw) {
	      return {
	        raw: '',
	        display: '',
	        type: 'unknown',
	        stateCode: null
	      };
	    }
	    const lower = raw.toLowerCase();
	    const hasRemote = REMOTE_HINTS.some(hint => lower.includes(hint));
	    const hasHybrid = HYBRID_HINTS.some(hint => lower.includes(hint));
	    const hasOnsite = ONSITE_HINTS.some(hint => lower.includes(hint));
	    let type = 'unknown';
	    if (hasHybrid) type = 'hybrid';
	    else if (hasOnsite) type = 'onsite';
	    else if (hasRemote) type = 'remote';

	    let display = cleanLocationLabel(raw);
	    if (!display) {
	      if (type === 'remote') display = 'Remote';
	      if (type === 'hybrid') display = 'Hybrid';
	      if (type === 'onsite') display = 'On-site';
	    }
	    if (type === 'remote' && display && !/\bremote\b/i.test(display)) {
	      display = `${display} (Remote)`;
	    }
	    if (type === 'hybrid' && display && !/\bhybrid\b/i.test(display)) {
	      display = `${display} (Hybrid)`;
	    }
	    const stateCode = type === 'remote'
	      ? null
	      : extractStateCodeFromText(display || raw) || extractStateCodeFromText(raw);
	    if (type === 'unknown' && stateCode) type = 'onsite';

	    return {
	      raw,
	      display: display || raw,
	      type,
	      stateCode
	    };
	  };

	  const formatLocationDisplay = (location = '') => parseLocation(location).display || '';

	  const isRemoteLocation = (location = '') => parseLocation(location).type === 'remote';

	  const extractStateCode = (location = '') => parseLocation(location).stateCode;

  const syncUnknownDate = (dateInput, unknownInput) => {
    if (!dateInput || !unknownInput) return;
    const isUnknown = Boolean(unknownInput.checked);
    dateInput.disabled = isUnknown;
    if (isUnknown) dateInput.value = '';
  };

  const initUnknownDateToggle = (dateInput, unknownInput, defaultUnknown = true) => {
    if (!dateInput || !unknownInput) return;
    if (defaultUnknown && !dateInput.value) {
      unknownInput.checked = true;
    }
    const sync = () => syncUnknownDate(dateInput, unknownInput);
    unknownInput.addEventListener('change', sync);
    sync();
  };

  const setUnknownDateValue = (dateInput, unknownInput, value, defaultUnknown = true) => {
    if (dateInput) dateInput.value = value || '';
    if (unknownInput) {
      unknownInput.checked = value ? false : defaultUnknown;
    }
    syncUnknownDate(dateInput, unknownInput);
  };

  const toTitle = (value) => value
    .toLowerCase()
    .split(' ')
    .filter(Boolean)
    .map(word => word[0].toUpperCase() + word.slice(1))
    .join(' ');

  const stripBom = (value) => value.replace(/^\uFEFF/, '');

  const normalizeHeader = (value) => stripBom(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');

  const parseCsv = (text) => {
    const rows = [];
    let row = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        row.push(field);
        field = '';
      } else if ((char === '\n' || char === '\r') && !inQuotes) {
        if (char === '\r' && text[i + 1] === '\n') i += 1;
        row.push(field);
        if (row.some(cell => cell.trim() !== '')) rows.push(row);
        row = [];
        field = '';
      } else {
        field += char;
      }
    }
    if (field.length || row.length) {
      row.push(field);
      if (row.some(cell => cell.trim() !== '')) rows.push(row);
    }
    return rows;
  };

  const CSV_HEADERS = {
    company: ['company', 'companyname', 'employer', 'organization'],
    title: ['title', 'role', 'position', 'jobtitle'],
    appliedDate: ['applieddate', 'dateapplied', 'applicationdate', 'applied', 'date'],
    postingDate: ['postingdate', 'posteddate', 'jobpostingdate', 'dateposted'],
    captureDate: ['capturedate', 'founddate', 'discoverdate', 'datefound', 'captured'],
    status: ['status', 'stage'],
    notes: ['notes', 'note', 'details'],
    jobUrl: ['joburl', 'url', 'link', 'joblink', 'applicationurl'],
    location: ['location', 'city', 'region', 'locale'],
    source: ['source', 'referral', 'channel', 'board'],
    batch: ['batch', 'batchname', 'importbatch'],
    tags: ['tags', 'tag', 'labels', 'label'],
    followUpDate: ['followupdate', 'followup', 'nextactiondate', 'followupdate'],
    followUpNote: ['followupnote', 'followupnotes', 'nextactionnote', 'nextactionnotes'],
    customFields: ['customfields', 'customfield', 'fields', 'metadata'],
    attachments: ['attachments', 'attachmentfiles', 'attachmentfile', 'files', 'documents'],
    resumeFile: ['resume', 'resumefile', 'resumefilename', 'resumeattachment', 'resumeattachmentname'],
    coverLetterFile: ['coverletter', 'coverletterfile', 'coverletterfilename', 'cover', 'coverfile']
  };

  const buildHeaderMap = (headers = []) => {
    const normalized = headers.map(header => normalizeHeader(header));
    const map = {};
    Object.entries(CSV_HEADERS).forEach(([key, aliases]) => {
      const idx = normalized.findIndex(value => aliases.includes(value));
      if (idx >= 0) map[key] = idx;
    });
    return map;
  };

  const parseCsvDate = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return '';
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return trimmed;
    const mdy = trimmed.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{2,4})$/);
    if (mdy) {
      const year = mdy[3].length === 2 ? `20${mdy[3]}` : mdy[3];
      const month = mdy[1].padStart(2, '0');
      const day = mdy[2].padStart(2, '0');
      const iso = `${year}-${month}-${day}`;
      return parseDateInput(iso) ? iso : '';
    }
    const parsed = new Date(trimmed);
    return Number.isNaN(parsed.getTime()) ? '' : formatDateInput(parsed);
  };
  const parseAttachmentList = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return [];
    return trimmed
      .split(/[;|]/)
      .map(item => item.trim())
      .filter(Boolean);
  };

  const parseTagList = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return [];
    return trimmed
      .split(/[;,|]/)
      .map(item => item.trim())
      .filter(Boolean);
  };

  const parseCustomFieldsInput = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return {};
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          return parsed;
        }
        if (Array.isArray(parsed)) {
          return parsed.reduce((acc, item) => {
            const key = (item?.key || '').toString().trim();
            const val = (item?.value || '').toString().trim();
            if (key && val) acc[key] = val;
            return acc;
          }, {});
        }
      } catch {}
    }
    const fields = {};
    trimmed.split(/[;|]/).forEach((pair) => {
      const parts = pair.split(':');
      if (!parts.length) return;
      const key = (parts.shift() || '').trim();
      const val = parts.join(':').trim();
      if (key && val) fields[key] = val;
    });
    return fields;
  };

  const normalizeUrl = (value) => {
    const trimmed = (value || '').toString().trim();
    if (!trimmed) return '';
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    return `https://${trimmed}`;
  };

  const readFileText = (file) => {
    if (file && typeof file.text === 'function') return file.text();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result || '');
      reader.onerror = () => reject(reader.error || new Error('Unable to read file.'));
      reader.readAsText(file);
    });
  };

  const parseJwt = (token) => {
    try {
      const payload = token.split('.')[1];
      if (!payload) return null;
      const normalized = payload.replace(/-/g, '+').replace(/_/g, '/');
      const decoded = atob(normalized.padEnd(normalized.length + (4 - normalized.length % 4) % 4, '='));
      return JSON.parse(decoded);
    } catch {
      return null;
    }
  };

  const getAuthClaims = (auth) => {
    if (!auth?.idToken) return {};
    return auth.claims || parseJwt(auth.idToken) || {};
  };

  const getAuthExpiresAt = (auth) => {
    const numeric = Number(auth?.expiresAt) || 0;
    if (numeric) return numeric;
    const claims = getAuthClaims(auth);
    if (claims?.exp) return claims.exp * 1000;
    return 0;
  };

  const normalizeAuth = (auth) => {
    if (!auth?.idToken) return null;
    const claims = getAuthClaims(auth);
    const expiresAt = getAuthExpiresAt({ ...auth, claims });
    return {
      ...auth,
      claims,
      expiresAt
    };
  };

  const loadAuth = () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(SHARED_STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.idToken) return null;
      return parsed;
    } catch {
      return null;
    }
  };

  const saveAuth = (auth) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(auth));
      localStorage.setItem(SHARED_STORAGE_KEY, JSON.stringify(auth));
    } catch {}
  };

  const clearAuth = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(SHARED_STORAGE_KEY);
    } catch {}
    state.auth = null;
  };

  const authIsValid = (auth) => {
    if (!auth || !auth.idToken) return false;
    const expiresAt = getAuthExpiresAt(auth);
    if (!expiresAt) return false;
    if (Date.now() > expiresAt - 60 * 1000) return false;
    return true;
  };

  const getCssColor = (value) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return value;
    ctx.fillStyle = value;
    return ctx.fillStyle;
  };

  const toRgba = (value, alpha) => {
    const normalized = getCssColor(value);
    if (normalized.startsWith('rgba(')) {
      return normalized.replace(/rgba\\(([^)]+),\\s*[^)]+\\)/, `rgba($1, ${alpha})`);
    }
    if (normalized.startsWith('rgb(')) {
      return normalized.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
    }
    return value;
  };

  const readCssVar = (name, fallback) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (value && value.trim()) ? value.trim() : fallback;
  };

  const joinUrl = (base, path) => {
    if (!base) return path;
    if (!path) return base;
    const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${cleanBase}${cleanPath}`;
  };

  const setStatus = (el, message, tone = '') => {
    if (!el) return;
    el.textContent = message;
    if (tone) {
      el.dataset.tone = tone;
    } else {
      delete el.dataset.tone;
    }
  };

  const setAuthMessage = (message, tone = '') => {
    authUiMessage = (message || '').toString();
    authUiTone = tone || '';
    if (els.authStatus) setStatus(els.authStatus, authUiMessage, authUiTone);
  };

  const clearAuthMessage = () => {
    authUiMessage = '';
    authUiTone = '';
    if (els.authStatus) setStatus(els.authStatus, '', '');
  };

  const setAuthModalBackgroundLocked = (locked) => {
    authModalBackgroundEls.forEach((el) => {
      if (locked) {
        if (!('jobtrackAuthHidden' in el.dataset)) {
          el.dataset.jobtrackAuthHidden = el.getAttribute('aria-hidden') || '';
        }
        el.setAttribute('aria-hidden', 'true');
        el.setAttribute('inert', '');
        return;
      }
      if (!('jobtrackAuthHidden' in el.dataset)) return;
      if (el.dataset.jobtrackAuthHidden) {
        el.setAttribute('aria-hidden', el.dataset.jobtrackAuthHidden);
      } else {
        el.removeAttribute('aria-hidden');
      }
      delete el.dataset.jobtrackAuthHidden;
      el.removeAttribute('inert');
    });
  };

  const AUTH_MODAL_FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex=\"-1\"])';

  const trapAuthModalFocus = (event) => {
    if (event.key !== 'Tab') return;
    if (!els.authModal || !els.authModal.classList.contains('active')) return;
    const content = els.authModal.querySelector('.modal-content');
    if (!content) return;
    const focusables = $$(AUTH_MODAL_FOCUSABLE, content).filter(el => el.offsetParent !== null);
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;

    if (!content.contains(active)) {
      event.preventDefault();
      first.focus({ preventScroll: true });
      return;
    }

    if (event.shiftKey && active === first) {
      event.preventDefault();
      last.focus({ preventScroll: true });
      return;
    }

    if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus({ preventScroll: true });
    }
  };

  const openAuthModal = () => {
    if (!els.authModal) return;
    if (els.authModal.classList.contains('active') && !els.authModal.hasAttribute('hidden')) return;
    authModalPrevFocus = document.activeElement;
    if (els.detailModal && els.detailModal.classList.contains('active')) {
      closeDetailModal();
    }
    els.authModal.removeAttribute('hidden');
    els.authModal.classList.add('active');
    document.body.classList.add('modal-open');
    setAuthModalBackgroundLocked(true);

    const content = els.authModal.querySelector('.modal-content');
    if (content && !authModalTrapBound) {
      content.addEventListener('keydown', trapAuthModalFocus);
      authModalTrapBound = true;
    }

    if (els.signIn && typeof els.signIn.focus === 'function') {
      els.signIn.focus({ preventScroll: true });
      return;
    }
    if (content && typeof content.focus === 'function') {
      content.focus({ preventScroll: true });
    }
  };

  const closeAuthModal = () => {
    if (!els.authModal) return;
    const content = els.authModal.querySelector('.modal-content');
    if (content && authModalTrapBound) {
      content.removeEventListener('keydown', trapAuthModalFocus);
      authModalTrapBound = false;
    }

    els.authModal.classList.remove('active');
    els.authModal.setAttribute('hidden', '');
    setAuthModalBackgroundLocked(false);
    if (!document.querySelector('.modal.active')) {
      document.body.classList.remove('modal-open');
    }
    if (authModalPrevFocus && typeof authModalPrevFocus.focus === 'function') {
      authModalPrevFocus.focus({ preventScroll: true });
    }
    authModalPrevFocus = null;
  };

  const startAuthWatcher = () => {
    if (authExpiryTimer) {
      window.clearTimeout(authExpiryTimer);
      authExpiryTimer = null;
    }
    if (!state.auth) return;
    const expiresAt = getAuthExpiresAt(state.auth);
    if (!expiresAt) return;
    const msUntil = Math.max(0, expiresAt - 60 * 1000 - Date.now());
    authExpiryTimer = window.setTimeout(() => {
      if (!state.auth) return;
      if (authIsValid(state.auth)) {
        startAuthWatcher();
        return;
      }
      clearAuth();
      setAuthMessage('Your session ended. Please sign in again.', 'info');
      updateAuthUI();
    }, msUntil);
  };

  const setImportProgress = (value, max, label) => {
    if (!els.importProgress || !els.importProgressWrap) return;
    const safeMax = Math.max(1, Number.isFinite(max) ? Math.floor(max) : 1);
    const safeValue = Math.max(0, Math.min(Number.isFinite(value) ? Math.floor(value) : 0, safeMax));
    els.importProgress.max = safeMax;
    els.importProgress.value = safeValue;
    els.importProgressWrap.dataset.state = 'visible';
    if (els.importProgressLabel && label) {
      els.importProgressLabel.textContent = label;
    }
  };

  const resetImportProgress = (label = 'Upload progress') => {
    if (!els.importProgress || !els.importProgressWrap) return;
    els.importProgressWrap.dataset.state = 'hidden';
    els.importProgress.max = 1;
    els.importProgress.value = 0;
    if (els.importProgressLabel) {
      els.importProgressLabel.textContent = label;
    }
  };
  const setProspectImportProgress = (value, max, label) => {
    if (!els.prospectImportProgress || !els.prospectImportProgressWrap) return;
    const safeMax = Number.isFinite(max) && max > 0 ? max : 1;
    const safeValue = Number.isFinite(value) ? Math.min(Math.max(value, 0), safeMax) : 0;
    els.prospectImportProgress.max = safeMax;
    els.prospectImportProgress.value = safeValue;
    els.prospectImportProgressWrap.dataset.state = 'visible';
    if (els.prospectImportProgressLabel && label) {
      els.prospectImportProgressLabel.textContent = label;
    }
  };
  const resetProspectImportProgress = (label = 'Import progress') => {
    if (!els.prospectImportProgress || !els.prospectImportProgressWrap) return;
    els.prospectImportProgressWrap.dataset.state = 'hidden';
    els.prospectImportProgress.max = 1;
    els.prospectImportProgress.value = 0;
    if (els.prospectImportProgressLabel) {
      els.prospectImportProgressLabel.textContent = label;
    }
  };

  const setOverlay = (el, message) => {
    if (!el) return;
    if (!message) {
      el.dataset.state = 'hidden';
      el.textContent = '';
      return;
    }
    el.dataset.state = '';
    el.textContent = message;
  };

  const getEntryLabel = (entry) => {
    const label = [entry?.title, entry?.company].filter(Boolean).join(' · ');
    return label || 'Entry';
  };

  const getEntryStatusLabel = (entry) => {
    const entryType = entry?.entryType || getEntryType(entry);
    const fallback = entryType === 'prospect' ? 'Active' : 'Applied';
    return toTitle((entry?.status || fallback).toString());
  };

  const sortEntriesByAppliedDate = (items = []) => {
    const sorted = [...items];
    sorted.sort((a, b) => {
      const aDate = parseDateInput(a?.appliedDate);
      const bDate = parseDateInput(b?.appliedDate);
      const aTime = aDate ? aDate.getTime() : 0;
      const bTime = bDate ? bDate.getTime() : 0;
      if (aTime !== bTime) return bTime - aTime;
      return getEntryLabel(a).localeCompare(getEntryLabel(b), 'en', { sensitivity: 'base' });
    });
    return sorted;
  };

	  const formatEntryMeta = (entry) => {
	    const parts = [];
	    const status = getEntryStatusLabel(entry);
	    if (status) parts.push(status);
	    const location = formatLocationDisplay(entry?.location || '');
	    if (location) parts.push(location);
	    const entryType = entry?.entryType || getEntryType(entry);
	    const dateValue = entryType === 'prospect' ? entry?.captureDate : entry?.appliedDate;
    const dateLabel = dateValue ? parseDateInput(dateValue) : null;
    if (dateLabel) {
      parts.push(`${entryType === 'prospect' ? 'Found' : 'Applied'} ${formatDateLabel(dateValue)}`);
    }
    const batch = (entry?.batch || '').toString().trim();
    if (batch) parts.push(`Batch: ${batch}`);
    const attachments = Array.isArray(entry?.attachments) ? entry.attachments : [];
    if (attachments.length) {
      parts.push(`${attachments.length} attachment${attachments.length === 1 ? '' : 's'}`);
    }
    return parts.filter(Boolean).join(' · ');
  };

  const getProspectSortDate = (entry) => {
    const posted = parseDateInput(entry?.postingDate);
    if (posted) return posted;
    const captured = parseDateInput(entry?.captureDate);
    if (captured) return captured;
    return parseIsoDate(entry?.createdAt) || parseIsoDate(entry?.updatedAt);
  };

	  const formatProspectMeta = (entry) => {
	    const parts = [];
	    const status = toTitle((entry?.status || 'Active').toString());
	    if (status) parts.push(status);
	    if (entry?.captureDate && parseDateInput(entry.captureDate)) {
	      parts.push(`Found ${formatDateLabel(entry.captureDate)}`);
	    }
	    const location = formatLocationDisplay(entry?.location || '');
	    if (location) parts.push(location);
	    const source = (entry?.source || '').toString().trim();
	    if (source) parts.push(source);
    const batch = (entry?.batch || '').toString().trim();
    if (batch) parts.push(`Batch: ${batch}`);
    return parts.filter(Boolean).join(' · ');
  };

  const getEntryStatusKey = (entry) => getEntryStatusLabel(entry).toLowerCase();

  const getEntryStatusGroup = (entry) => {
    const key = getEntryStatusKey(entry);
    if (STATUS_GROUPS.rejected.has(key)) return 'rejected';
    if (STATUS_GROUPS.archived.has(key)) return 'archived';
    if (STATUS_GROUPS.active.has(key)) return 'active';
    return 'active';
  };

  const getStatusTone = (statusKey = '') => {
    const key = (statusKey || '').toString().trim().toLowerCase();
    if (['applied', 'screening', 'interview', 'offer', 'rejected', 'withdrawn', 'active', 'interested', 'inactive'].includes(key)) {
      return key;
    }
    return 'applied';
  };

  const getDayKey = (date) => Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());

  const addDays = (value, days) => {
    if (!value && value !== 0) return null;
    const parsed = value instanceof Date ? value : (parseDateInput(value) || parseIsoDate(value));
    if (!parsed) return null;
    const due = new Date(parsed.getTime());
    due.setUTCDate(due.getUTCDate() + days);
    return due;
  };

  const buildNextAction = (verb, dueDate) => {
    if (!verb) return { label: 'Follow up soon', tone: '' };
    if (!dueDate) return { label: `${verb} soon`, tone: '' };
    const dayMs = 24 * 60 * 60 * 1000;
    const todayKey = getDayKey(new Date());
    const dueKey = getDayKey(dueDate);
    const diffDays = Math.round((dueKey - todayKey) / dayMs);
    const dateLabel = formatDateLabel(formatDateInput(dueDate));
    if (diffDays <= 0) {
      return { label: `${verb} now`, tone: 'danger' };
    }
    if (diffDays <= 2) {
      return { label: `${verb} by ${dateLabel}`, tone: 'warning' };
    }
    return { label: `${verb} by ${dateLabel}`, tone: '' };
  };

  const getNextAction = (entry) => {
    const entryType = entry?.entryType || getEntryType(entry);
    const statusKey = getEntryStatusKey(entry);
    const followUpDate = parseDateInput(entry?.followUpDate);
    if (entryType === 'prospect') {
      if (statusKey === 'rejected') return { label: 'Closed', tone: 'muted' };
      if (statusKey === 'inactive') return { label: 'Archived', tone: 'muted' };
      const verb = statusKey === 'interested' ? 'Apply' : 'Review';
      if (followUpDate) {
        return buildNextAction(verb, followUpDate);
      }
      const baseDate = entry.captureDate || deriveStatusDate(entry);
      const offset = FOLLOW_UP_DAYS[statusKey] ?? 5;
      return buildNextAction(verb, addDays(baseDate, offset));
    }
    if (statusKey === 'offer') return { label: 'Review offer', tone: 'success' };
    if (statusKey === 'rejected' || statusKey === 'withdrawn') return { label: 'Closed', tone: 'muted' };
    if (followUpDate) {
      return buildNextAction('Follow up', followUpDate);
    }
    const baseDate = deriveStatusDate(entry) || entry.appliedDate;
    const offset = FOLLOW_UP_DAYS[statusKey] ?? 7;
    return buildNextAction('Follow up', addDays(baseDate, offset));
  };

  const buildDetailEntryList = (entries = []) => {
    const wrap = document.createElement('div');
    wrap.className = 'jobtrack-detail-entries';

    const head = document.createElement('div');
    head.className = 'jobtrack-detail-entries-head';
    const label = document.createElement('span');
    label.textContent = 'Entries';
    const count = document.createElement('span');
    count.textContent = `${entries.length} total`;
    head.appendChild(label);
    head.appendChild(count);
    wrap.appendChild(head);

    const list = document.createElement('ul');
    list.className = 'jobtrack-detail-entry-list';
    entries.forEach((entry) => {
      const entryLabel = getEntryLabel(entry);
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'jobtrack-detail-entry';
      button.dataset.jobtrackDetailEntry = entry?.applicationId || '';
      button.setAttribute('aria-label', `View ${entryLabel}`);

      const title = document.createElement('span');
      title.className = 'jobtrack-detail-entry-title';
      title.textContent = entryLabel;
      const meta = document.createElement('span');
      meta.className = 'jobtrack-detail-entry-meta';
      meta.textContent = formatEntryMeta(entry) || 'View entry details';

      button.appendChild(title);
      button.appendChild(meta);
      if (!entry?.applicationId) {
        button.disabled = true;
        button.setAttribute('aria-disabled', 'true');
        button.title = 'Entry details unavailable';
      }
      item.appendChild(button);
      list.appendChild(item);
    });
    wrap.appendChild(list);
    return wrap;
  };

  const setDashboardDetail = (title, lines = [], entries = []) => {
    if (els.detailSubtitle) {
      els.detailSubtitle.textContent = title || DETAIL_DEFAULT_SUBTITLE;
    }
    if (!els.detailBody) return;
    els.detailBody.innerHTML = '';
    if (!lines.length && !entries.length) {
      els.detailBody.textContent = DETAIL_DEFAULT_BODY;
      return;
    }
    if (lines.length) {
      const list = document.createElement('ul');
      list.className = 'jobtrack-detail-list';
      lines.forEach((line) => {
        const item = document.createElement('li');
        item.textContent = line;
        list.appendChild(item);
      });
      els.detailBody.appendChild(list);
    }
    if (entries.length) {
      const sorted = sortEntriesByAppliedDate(entries);
      els.detailBody.appendChild(buildDetailEntryList(sorted));
    }
  };

  const clearDashboardDetail = () => {
    if (els.detailSubtitle) els.detailSubtitle.textContent = DETAIL_DEFAULT_SUBTITLE;
    if (els.detailBody) els.detailBody.textContent = DETAIL_DEFAULT_BODY;
  };

  const buildDetailModalRow = (label, value, href = '') => {
    if (!value) return null;
    const row = document.createElement('div');
    row.className = 'jobtrack-modal-meta-row';
    const labelEl = document.createElement('span');
    labelEl.className = 'jobtrack-modal-meta-label';
    labelEl.textContent = label;
    const valueEl = document.createElement(href ? 'a' : 'span');
    valueEl.className = 'jobtrack-modal-meta-value';
    valueEl.textContent = value;
    if (href) {
      valueEl.href = href;
      valueEl.target = '_blank';
      valueEl.rel = 'noopener';
    }
    row.appendChild(labelEl);
    row.appendChild(valueEl);
    return row;
  };

  const buildStatusHistoryList = (entry) => {
    const history = Array.isArray(entry?.statusHistory) ? entry.statusHistory : [];
    if (!history.length) return null;
    const entryType = entry?.entryType || getEntryType(entry);
    const appliedDateValue = entryType === 'application' && entry?.appliedDate && parseDateInput(entry.appliedDate)
      ? entry.appliedDate
      : '';
    let items = history
      .map(item => {
        const statusRaw = (item?.status || '').toString().trim();
        return {
          status: toTitle(statusRaw),
          statusLower: statusRaw.toLowerCase(),
          date: parseHistoryDate(item?.date)
        };
      })
      .filter(item => item.date);
    if (appliedDateValue) {
      items = items.filter(item => item.statusLower !== 'applied');
    }
    items.sort((a, b) => b.date - a.date);
    if (!items.length) return null;
    const wrap = document.createElement('div');
    wrap.className = 'jobtrack-modal-history';
    const title = document.createElement('p');
    title.className = 'jobtrack-modal-attachments-title';
    title.textContent = 'Status history';
    const list = document.createElement('ul');
    list.className = 'jobtrack-modal-history-list';
    items.forEach((item) => {
      const row = document.createElement('li');
      row.className = 'jobtrack-modal-history-item';
      const status = document.createElement('span');
      status.className = 'jobtrack-modal-history-status';
      status.textContent = item.status || 'Status update';
      const date = document.createElement('span');
      date.className = 'jobtrack-modal-history-date';
      date.textContent = formatDateLabel(formatDateInput(item.date));
      row.appendChild(status);
      row.appendChild(date);
      list.appendChild(row);
    });
    wrap.appendChild(title);
    wrap.appendChild(list);
    return wrap;
  };

  const buildProspectApplyForm = (entry) => {
    if (!entry?.applicationId) return null;
    const wrap = document.createElement('div');
    wrap.className = 'jobtrack-modal-apply-form';
    const title = document.createElement('p');
    title.className = 'jobtrack-modal-attachments-title';
    title.textContent = 'Convert to application';
    wrap.appendChild(title);

    const fields = document.createElement('div');
    fields.className = 'jobtrack-modal-apply-fields';
    const dateField = document.createElement('div');
    dateField.className = 'jobtrack-field';
    const dateLabel = document.createElement('label');
    const dateId = `jobtrack-modal-apply-date-${entry.applicationId.toString().replace(/[^a-z0-9_-]/gi, '') || 'prospect'}`;
    dateLabel.className = 'jobtrack-label';
    dateLabel.setAttribute('for', dateId);
    dateLabel.textContent = 'Applied date';
    const dateInput = document.createElement('input');
    dateInput.type = 'date';
    dateInput.className = 'jobtrack-input';
    dateInput.id = dateId;
    dateInput.dataset.jobtrack = 'apply-date-input';
    const fallbackDate = entry.captureDate && parseDateInput(entry.captureDate)
      ? entry.captureDate
      : formatDateInput(new Date());
    dateInput.value = fallbackDate;
    dateField.appendChild(dateLabel);
    dateField.appendChild(dateInput);
    fields.appendChild(dateField);

    const applyBtn = document.createElement('button');
    applyBtn.type = 'button';
    applyBtn.className = 'btn-primary jobtrack-modal-apply-btn';
    applyBtn.textContent = 'Convert and apply';
    applyBtn.addEventListener('click', async () => {
      const ok = await applyProspect(entry.applicationId, dateInput.value, els.detailModalStatus);
      if (ok) closeDetailModal();
    });
    fields.appendChild(applyBtn);
    wrap.appendChild(fields);
    const help = document.createElement('p');
    help.className = 'jobtrack-help';
    help.textContent = 'Creates an application entry and removes this prospect.';
    wrap.appendChild(help);
    return wrap;
  };

  const renderDetailModal = (entry) => {
    if (!els.detailModalBody) return;
    els.detailModalBody.innerHTML = '';
    if (els.detailModalStatus) setStatus(els.detailModalStatus, '', '');
    if (!entry) {
      els.detailModalBody.textContent = 'Entry details unavailable.';
      return;
    }
    const entryType = entry.entryType || getEntryType(entry);
    const statusValue = (entry.status || '').toString().trim().toLowerCase();

    const title = entry.title || 'Entry details';
    if (els.detailModalTitle) els.detailModalTitle.textContent = title;
    if (els.detailModalSubtitle) {
      const subtitleParts = [entry.company, getEntryStatusLabel(entry)];
      if (entryType === 'prospect') {
        if (entry.captureDate) subtitleParts.push(`Found ${formatDateLabel(entry.captureDate)}`);
      } else if (entry.appliedDate) {
        subtitleParts.push(`Applied ${formatDateLabel(entry.appliedDate)}`);
      }
      els.detailModalSubtitle.textContent = subtitleParts.filter(Boolean).join(' · ');
    }

    const appliedValue = entry.appliedDate && parseDateInput(entry.appliedDate)
      ? entry.appliedDate
      : '';
    const appliedLabel = appliedValue ? formatDateLabel(appliedValue) : '';
    const captureLabel = entry.captureDate && parseDateInput(entry.captureDate)
      ? formatDateLabel(entry.captureDate)
      : '';
    const postingLabel = entry.postingDate && parseDateInput(entry.postingDate)
      ? formatDateLabel(entry.postingDate)
      : '';
    const statusDateValue = entry.statusDate && parseDateInput(entry.statusDate)
      ? entry.statusDate
      : '';
    const statusDateLabel = statusDateValue ? formatDateLabel(statusDateValue) : '';
    const showStatusDate = Boolean(statusDateLabel)
      && !(entryType === 'application' && statusValue === 'applied' && appliedValue);
    const tagsLabel = Array.isArray(entry.tags) && entry.tags.length ? entry.tags.join(', ') : '';
    const followUpLabel = entry.followUpDate && parseDateInput(entry.followUpDate)
      ? formatDateLabel(entry.followUpDate)
      : '';
    const customFieldLabel = entry.customFields && typeof entry.customFields === 'object'
      ? Object.entries(entry.customFields).map(([key, value]) => `${key}: ${value}`).join(' · ')
      : '';

    const meta = document.createElement('div');
    meta.className = 'jobtrack-modal-meta';
    const rows = [
      buildDetailModalRow('Company', entry.company || ''),
      buildDetailModalRow('Status', getEntryStatusLabel(entry)),
      showStatusDate ? buildDetailModalRow('Status date', statusDateLabel) : null,
      buildDetailModalRow('Applied date', appliedLabel),
      buildDetailModalRow('Found date', captureLabel),
      buildDetailModalRow('Posted', postingLabel),
      buildDetailModalRow('Location', formatLocationDisplay(entry.location || '')),
      buildDetailModalRow('Source', entry.source || ''),
      buildDetailModalRow('Batch', entry.batch || ''),
      buildDetailModalRow('Tags', tagsLabel),
      buildDetailModalRow('Follow-up', followUpLabel),
      buildDetailModalRow('Follow-up note', entry.followUpNote || ''),
      buildDetailModalRow('Custom fields', customFieldLabel),
      buildDetailModalRow('Job URL', entry.jobUrl || '', entry.jobUrl ? normalizeUrl(entry.jobUrl) : '')
    ].filter(Boolean);
    rows.forEach((row) => meta.appendChild(row));
    if (rows.length) els.detailModalBody.appendChild(meta);

    if (entry.notes) {
      const notesWrap = document.createElement('div');
      const notesTitle = document.createElement('p');
      notesTitle.className = 'jobtrack-modal-attachments-title';
      notesTitle.textContent = 'Notes';
      const notes = document.createElement('p');
      notes.className = 'jobtrack-modal-notes';
      notes.textContent = entry.notes;
      notesWrap.appendChild(notesTitle);
      notesWrap.appendChild(notes);
      els.detailModalBody.appendChild(notesWrap);
    }

    const historyWrap = buildStatusHistoryList(entry);
    if (historyWrap) els.detailModalBody.appendChild(historyWrap);

    const actionsWrap = document.createElement('div');
    actionsWrap.className = 'jobtrack-modal-actions';

    const actionRow = document.createElement('div');
    actionRow.className = 'jobtrack-modal-action-row';
    if (entry.applicationId) {
      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'btn-ghost';
      editBtn.textContent = 'Edit entry';
      editBtn.addEventListener('click', () => {
        setEntryEditMode(entry);
        closeDetailModal();
      });
      actionRow.appendChild(editBtn);

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'btn-ghost';
      deleteBtn.textContent = 'Delete entry';
      deleteBtn.addEventListener('click', async () => {
        await deleteEntry(entry.applicationId);
        closeDetailModal();
      });
      actionRow.appendChild(deleteBtn);
    }
    if (actionRow.childNodes.length) actionsWrap.appendChild(actionRow);

    if (entry.applicationId) {
      const statusWrap = document.createElement('div');
      statusWrap.className = 'jobtrack-modal-status-form';
      const statusTitle = document.createElement('p');
      statusTitle.className = 'jobtrack-modal-attachments-title';
      statusTitle.textContent = 'Update status';
      statusWrap.appendChild(statusTitle);

      const statusFields = document.createElement('div');
      statusFields.className = 'jobtrack-modal-status-fields';
      const idSuffix = entry.applicationId.toString().replace(/[^a-z0-9_-]/gi, '') || 'entry';

      const statusField = document.createElement('div');
      statusField.className = 'jobtrack-field';
      const statusLabel = document.createElement('label');
      const statusId = `jobtrack-modal-status-${idSuffix}`;
      statusLabel.className = 'jobtrack-label';
      statusLabel.setAttribute('for', statusId);
      statusLabel.textContent = 'Status';
      const statusSelect = document.createElement('select');
      statusSelect.className = 'jobtrack-select';
      statusSelect.id = statusId;
      const statusOptions = entryType === 'prospect' ? PROSPECT_STATUSES : APPLICATION_STATUSES;
      const currentStatus = toTitle((entry.status || (entryType === 'prospect' ? 'Active' : 'Applied')).toString());
      const optionList = statusOptions.includes(currentStatus)
        ? statusOptions
        : [currentStatus, ...statusOptions];
      optionList.forEach((option) => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        statusSelect.appendChild(opt);
      });
      statusSelect.value = currentStatus;
      statusField.appendChild(statusLabel);
      statusField.appendChild(statusSelect);
      statusFields.appendChild(statusField);

      let statusDateInput = null;
      const allowStatusDate = entryType === 'application' || entryType === 'prospect';
      if (allowStatusDate) {
        const dateField = document.createElement('div');
        dateField.className = 'jobtrack-field';
        const dateLabel = document.createElement('label');
        const dateId = `jobtrack-modal-status-date-${idSuffix}`;
        dateLabel.className = 'jobtrack-label';
        dateLabel.setAttribute('for', dateId);
        dateLabel.textContent = entryType === 'application' ? 'Status date' : 'Status date (optional)';
        statusDateInput = document.createElement('input');
        statusDateInput.type = 'date';
        statusDateInput.className = 'jobtrack-input';
        statusDateInput.id = dateId;
        if (entry.statusDate && parseDateInput(entry.statusDate)) {
          statusDateInput.value = entry.statusDate;
        } else if (entryType === 'application') {
          statusDateInput.value = formatDateInput(new Date());
        } else {
          statusDateInput.value = '';
        }
        dateField.appendChild(dateLabel);
        dateField.appendChild(statusDateInput);
        statusFields.appendChild(dateField);
      }

      const updateBtn = document.createElement('button');
      updateBtn.type = 'button';
      updateBtn.className = 'btn-primary jobtrack-modal-status-btn';
      updateBtn.textContent = 'Save status';
      updateBtn.addEventListener('click', () => {
        updateEntryStatus(entry, statusSelect.value, statusDateInput?.value || '');
      });
      statusFields.appendChild(updateBtn);
      statusWrap.appendChild(statusFields);
      actionsWrap.appendChild(statusWrap);
    }

    if (entryType === 'prospect') {
      const applyWrap = buildProspectApplyForm(entry);
      if (applyWrap) actionsWrap.appendChild(applyWrap);
    }

    if (actionsWrap.childNodes.length) {
      els.detailModalBody.appendChild(actionsWrap);
    }

    const attachments = Array.isArray(entry.attachments) ? entry.attachments : [];
    const attachmentWrap = document.createElement('div');
    attachmentWrap.className = 'jobtrack-modal-attachments';
    const attachmentHead = document.createElement('div');
    attachmentHead.className = 'jobtrack-modal-attachments-head';
    const attachmentTitle = document.createElement('p');
    attachmentTitle.className = 'jobtrack-modal-attachments-title';
    attachmentTitle.textContent = 'Attachments';
    attachmentHead.appendChild(attachmentTitle);

    if (entry.applicationId) {
      const zipBtn = document.createElement('button');
      zipBtn.type = 'button';
      zipBtn.className = 'btn-ghost jobtrack-modal-attachment-btn';
      zipBtn.textContent = 'Download ZIP';
      zipBtn.setAttribute('aria-label', 'Download all attachments as ZIP');
      if (!attachments.length) {
        zipBtn.disabled = true;
        zipBtn.setAttribute('aria-disabled', 'true');
        zipBtn.title = 'No attachments to zip';
      } else {
        zipBtn.addEventListener('click', () => downloadEntryZip(entry.applicationId, els.detailModalStatus));
      }
      attachmentHead.appendChild(zipBtn);
    }
    attachmentWrap.appendChild(attachmentHead);

    if (!attachments.length) {
      const empty = document.createElement('p');
      empty.className = 'jobtrack-form-status';
      empty.textContent = 'No attachments uploaded yet.';
      attachmentWrap.appendChild(empty);
    } else {
      const list = document.createElement('ul');
      list.className = 'jobtrack-modal-attachment-list';
      attachments.forEach((attachment, index) => {
        const row = document.createElement('li');
        row.className = 'jobtrack-modal-attachment';

        const info = document.createElement('div');
        info.className = 'jobtrack-modal-attachment-info';
        const name = document.createElement('div');
        name.className = 'jobtrack-modal-attachment-name';
        name.textContent = attachment?.filename || `Attachment ${index + 1}`;
        info.appendChild(name);

        const metaInfo = document.createElement('div');
        metaInfo.className = 'jobtrack-modal-attachment-meta';
        if (attachment?.kind) {
          const kind = document.createElement('span');
          kind.textContent = toTitle(attachment.kind.replace(/-/g, ' '));
          metaInfo.appendChild(kind);
        }
        if (attachment?.uploadedAt) {
          const uploaded = document.createElement('span');
          uploaded.textContent = formatDateLabel(attachment.uploadedAt);
          metaInfo.appendChild(uploaded);
        }
        if (attachment?.size) {
          const size = document.createElement('span');
          size.textContent = formatFileSize(Number(attachment.size));
          metaInfo.appendChild(size);
        }
        if (metaInfo.childNodes.length) info.appendChild(metaInfo);

        const downloadBtn = document.createElement('button');
        downloadBtn.type = 'button';
        downloadBtn.className = 'btn-ghost jobtrack-modal-attachment-btn';
        downloadBtn.textContent = 'Download';
        downloadBtn.setAttribute('aria-label', `Download ${name.textContent}`);
        if (!attachment?.key) {
          downloadBtn.disabled = true;
          downloadBtn.setAttribute('aria-disabled', 'true');
          downloadBtn.title = 'Attachment unavailable';
        } else {
          downloadBtn.addEventListener('click', () => downloadAttachment(attachment, els.detailModalStatus));
        }

        row.appendChild(info);
        row.appendChild(downloadBtn);
        list.appendChild(row);
      });
      attachmentWrap.appendChild(list);
    }
    els.detailModalBody.appendChild(attachmentWrap);
  };

  const openDetailModal = (entry, focusSelector = '') => {
    if (!els.detailModal) return;
    state.detailModalEntryId = entry?.applicationId || null;
    state.detailModalPrevFocus = document.activeElement;
    renderDetailModal(entry);
    els.detailModal.classList.add('active');
    document.body.classList.add('modal-open');
    const content = els.detailModal.querySelector('.modal-content');
    const focusTarget = focusSelector ? els.detailModal.querySelector(focusSelector) : null;
    if (focusTarget && typeof focusTarget.focus === 'function') {
      focusTarget.focus({ preventScroll: true });
      return;
    }
    if (content && typeof content.focus === 'function') {
      content.focus({ preventScroll: true });
    }
  };

  const closeDetailModal = () => {
    if (!els.detailModal) return;
    els.detailModal.classList.remove('active');
    if (!document.querySelector('.modal.active')) {
      document.body.classList.remove('modal-open');
    }
    if (state.detailModalPrevFocus && typeof state.detailModalPrevFocus.focus === 'function') {
      state.detailModalPrevFocus.focus();
    }
    state.detailModalPrevFocus = null;
    state.detailModalEntryId = null;
  };

  const confirmAction = (message) => {
    if (typeof window === 'undefined' || typeof window.confirm !== 'function') return true;
    return window.confirm(message);
  };

  const promptActionNote = (message, defaultValue = '') => {
    if (typeof window === 'undefined' || typeof window.prompt !== 'function') return '';
    const response = window.prompt(message, defaultValue);
    if (response === null || response === undefined) return '';
    return response.toString().trim();
  };

  const appendEntryNote = (existingNotes, noteLine) => {
    const base = (existingNotes || '').toString().trim();
    const next = (noteLine || '').toString().trim();
    if (!next) return base;
    if (!base) return next;
    return `${base}\n${next}`;
  };

  const updateConfigStatus = () => {
    if (els.apiStatus) {
      els.apiStatus.textContent = config.apiBase ? 'Configured' : 'Not configured';
    }
    if (els.cognitoStatus) {
      els.cognitoStatus.textContent = (config.cognitoDomain && config.cognitoClientId && config.cognitoRedirect)
        ? 'Configured'
        : 'Not configured';
    }
  };

  const updateEntrySummary = (items = []) => {
    if (!els.entrySummary) return;
    if (!authIsValid(state.auth)) {
      els.entrySummary.textContent = 'Sign in to view summary.';
      return;
    }
    if (!items.length) {
      els.entrySummary.textContent = 'No activity yet.';
      return;
    }
    let active = 0;
    let interviews = 0;
    let offers = 0;
    items.forEach((entry) => {
      const entryType = entry?.entryType || getEntryType(entry);
      const statusKey = getEntryStatusKey(entry);
      if (STATUS_GROUPS.active.has(statusKey)) active += 1;
      if (entryType === 'application' && statusKey === 'interview') interviews += 1;
      if (entryType === 'application' && statusKey === 'offer') offers += 1;
    });
    els.entrySummary.textContent = `${active} active · ${interviews} interviews · ${offers} offers`;
  };

  const storeEntries = (items = []) => {
    state.entries = items;
    state.entryItems = new Map();
    items.forEach((item) => {
      if (item && item.applicationId) state.entryItems.set(item.applicationId, item);
    });
    updateEntrySummary(items);
  };

  const getEntryType = (item) => (item && item.recordType === 'prospect' ? 'prospect' : 'application');

  const updateEntrySubmitLabel = () => {
    if (!els.entrySubmit) return;
    const label = state.entryType === 'prospect' ? 'prospect' : 'application';
    els.entrySubmit.textContent = state.editingEntryId ? `Update ${label}` : `Save ${label}`;
  };

  const toggleEntryGroup = (nodes = [], isVisible) => {
    nodes.forEach((node) => {
      node.hidden = !isVisible;
      const inputs = node.querySelectorAll('input,select,textarea,button');
      inputs.forEach((input) => {
        input.disabled = !isVisible;
      });
    });
  };

  const updateStatusOptions = (type, preserveSelection = true) => {
    if (!els.statusInput) return;
    const options = type === 'prospect' ? PROSPECT_STATUSES : APPLICATION_STATUSES;
    const current = preserveSelection ? els.statusInput.value : '';
    els.statusInput.innerHTML = '';
    options.forEach((option) => {
      const opt = document.createElement('option');
      opt.value = option;
      opt.textContent = option;
      els.statusInput.appendChild(opt);
    });
    if (current && options.includes(current)) {
      els.statusInput.value = current;
    } else {
      els.statusInput.value = options[0];
    }
  };

  const setEntryType = (type, { preserveStatus = true } = {}) => {
    const nextType = type === 'prospect' ? 'prospect' : 'application';
    state.entryType = nextType;
    if (els.entryForm) els.entryForm.dataset.entryType = nextType;
    if (els.entryTypeInputs.length) {
      els.entryTypeInputs.forEach((input) => {
        input.checked = input.value === nextType;
      });
    }
    toggleEntryGroup(els.entryApplicationFields, nextType === 'application');
    toggleEntryGroup(els.entryProspectFields, nextType === 'prospect');
    if (els.jobUrlInput) els.jobUrlInput.required = nextType === 'prospect';
    if (els.appliedDateInput) els.appliedDateInput.required = nextType === 'application';
    if (els.appliedRequired) els.appliedRequired.hidden = nextType !== 'application';
    if (els.captureDateInput) els.captureDateInput.required = nextType === 'prospect';
    if (els.jobUrlHelp) {
      const label = nextType === 'prospect' ? 'Required for prospects.' : 'Optional for applications.';
      els.jobUrlHelp.textContent = `${label} Paste a job URL to auto-fill company and role.`;
    }
    if (els.captureLabel) {
      els.captureLabel.textContent = nextType === 'prospect' ? 'Found date' : 'Found date (optional)';
    }
    if (els.captureHelp) {
      els.captureHelp.textContent = nextType === 'prospect'
        ? 'Defaults to today for prospects.'
        : 'When you first saved or discovered the role.';
    }
    updateStatusOptions(nextType, preserveStatus);
    if (nextType === 'prospect') clearAttachmentInputs();
    updateEntrySubmitLabel();
  };

  const setEntryEditMode = (item) => {
    if (!item) return;
    const entryType = getEntryType(item);
    state.editingEntryId = item.applicationId || null;
    state.editingEntry = item;
    setEntryType(entryType, { preserveStatus: false });
    if (els.companyInput) els.companyInput.value = item.company || '';
    if (els.titleInput) els.titleInput.value = item.title || '';
    if (els.jobUrlInput) els.jobUrlInput.value = item.jobUrl || '';
    if (els.locationInput) els.locationInput.value = item.location || '';
    if (els.sourceInput) els.sourceInput.value = item.source || '';
    if (els.batchInput) els.batchInput.value = item.batch || '';
    if (els.appliedDateInput) els.appliedDateInput.value = item.appliedDate || '';
    if (els.captureDateInput) {
      els.captureDateInput.value = item.captureDate || '';
    }
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, item.postingDate || '');
    if (els.statusInput) els.statusInput.value = item.status || (entryType === 'prospect' ? 'Active' : 'Applied');
    if (els.notesInput) els.notesInput.value = item.notes || '';
    if (els.tagsInput) els.tagsInput.value = formatTagInput(item.tags || []);
    if (els.followUpDateInput) els.followUpDateInput.value = item.followUpDate || '';
    if (els.followUpNoteInput) els.followUpNoteInput.value = item.followUpNote || '';
    setCustomFieldsFromEntry(item.customFields || {});
    updateEntrySubmitLabel();
    setDraftStatus('');
    if (els.entryFormStatus) {
      const label = [item.title, item.company].filter(Boolean).join(' · ') || 'entry';
      setStatus(els.entryFormStatus, `Editing ${label}. Save to update or clear to cancel.`, 'info');
    }
    activateTab('entry', true);
  };

  const clearEntryEditMode = (message = 'Ready to save entries.', tone = '') => {
    state.editingEntryId = null;
    state.editingEntry = null;
    updateEntrySubmitLabel();
    if (els.entryFormStatus && message) setStatus(els.entryFormStatus, message, tone);
  };

  const activateTab = (name, shouldFocus = false) => {
    if (!tabs.buttons.length || !tabs.panels.length) return;
    const authed = authIsValid(state.auth);
    const safeName = authed || name === 'account' ? name : 'account';
    tabs.buttons.forEach((button) => {
      const selected = button.dataset.jobtrackTab === safeName;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    tabs.panels.forEach((panel) => {
      panel.hidden = panel.dataset.jobtrackPanel !== safeName;
    });
    if (shouldFocus) {
      const activeButton = tabs.buttons.find(button => button.dataset.jobtrackTab === safeName);
      if (activeButton) activeButton.focus();
    }
    if (safeName === 'dashboard') {
      window.requestAnimationFrame(() => {
        if (state.lineChart) state.lineChart.resize();
        if (state.statusChart) state.statusChart.resize();
      });
    }
  };

  const initTabs = () => {
    if (!tabs.buttons.length || !tabs.panels.length) return;
    const defaultTab = tabs.buttons.find(button => button.getAttribute('aria-selected') === 'true')?.dataset.jobtrackTab
      || tabs.buttons[0].dataset.jobtrackTab;
    const hash = window.location && window.location.hash
      ? window.location.hash.replace('#', '')
      : '';
    const initial = tabs.buttons.some(button => button.dataset.jobtrackTab === hash) ? hash : defaultTab;
    activateTab(initial);
    tabs.buttons.forEach((button, index) => {
      button.addEventListener('click', () => activateTab(button.dataset.jobtrackTab, true));
      button.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          const next = (index + 1) % tabs.buttons.length;
          activateTab(tabs.buttons[next].dataset.jobtrackTab, true);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          const prev = (index - 1 + tabs.buttons.length) % tabs.buttons.length;
          activateTab(tabs.buttons[prev].dataset.jobtrackTab, true);
        } else if (event.key === 'Home') {
          event.preventDefault();
          activateTab(tabs.buttons[0].dataset.jobtrackTab, true);
        } else if (event.key === 'End') {
          event.preventDefault();
          activateTab(tabs.buttons[tabs.buttons.length - 1].dataset.jobtrackTab, true);
        }
      });
    });
  };

  const initJumpButtons = () => {
    if (els.jumpEntryButtons.length) {
      els.jumpEntryButtons.forEach((button) => {
        button.addEventListener('click', () => {
          activateTab('entry', true);
          if (els.entryForm && typeof els.entryForm.scrollIntoView === 'function') {
            els.entryForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      });
    }
    if (els.jumpTabButtons.length) {
      els.jumpTabButtons.forEach((button) => {
        button.addEventListener('click', () => {
          const target = button.dataset.jobtrackJump;
          if (!target) return;
          activateTab(target, true);
          if (target === 'entry' && els.entryForm && typeof els.entryForm.scrollIntoView === 'function') {
            els.entryForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      });
    }
  };

  const randomBase64Url = (size = 32) => {
    const buffer = new Uint8Array(size);
    crypto.getRandomValues(buffer);
    const binary = String.fromCharCode(...buffer);
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  };

  const sha256 = async (plain) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(plain);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(digest);
  };

  const buildAuthorizeUrl = async () => {
    const verifier = randomBase64Url(48);
    const challengeBytes = await sha256(verifier);
    const challenge = btoa(String.fromCharCode(...challengeBytes))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    const authState = randomBase64Url(16);
    sessionStorage.setItem(STATE_KEY, authState);
    sessionStorage.setItem(VERIFIER_KEY, verifier);

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      scope: config.cognitoScopes,
      code_challenge_method: 'S256',
      code_challenge: challenge,
      state: authState
    });
    return `https://${config.cognitoDomain}/oauth2/authorize?${params.toString()}`;
  };

  const exchangeCodeForTokens = async (code) => {
    const verifier = sessionStorage.getItem(VERIFIER_KEY) || '';
    sessionStorage.removeItem(VERIFIER_KEY);
    sessionStorage.removeItem(STATE_KEY);
    if (!verifier) throw new Error('Missing PKCE verifier.');

    const params = new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      code,
      code_verifier: verifier
    });
    const res = await fetch(`https://${config.cognitoDomain}/oauth2/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params.toString()
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Unable to exchange auth code.');
    }
    const data = await res.json();
    if (!data.id_token) throw new Error('Missing id_token from auth response.');
    const claims = parseJwt(data.id_token) || {};
    const expiresAt = claims.exp ? claims.exp * 1000 : Date.now() + (data.expires_in || 3600) * 1000;
    const auth = {
      idToken: data.id_token,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt,
      claims
    };
    saveAuth(auth);
    state.auth = auth;
    return auth;
  };

	  const handleAuthRedirect = async () => {
	    const params = new URLSearchParams(window.location.search);
	    const code = params.get('code');
	    const returnedState = params.get('state') || '';
	    const storedState = sessionStorage.getItem(STATE_KEY) || '';
	    if (!code) return false;
	    if (storedState && returnedState && storedState !== returnedState) {
	      throw new Error('Auth state mismatch.');
	    }
	    setStatus(els.authStatus, 'Finalizing sign-in...', 'info');
	    await exchangeCodeForTokens(code);
	    params.delete('code');
	    params.delete('state');
	    const next = params.toString();
	    const nextUrl = next ? `${window.location.pathname}?${next}` : window.location.pathname;
	    window.history.replaceState({}, document.title, nextUrl);
	    const returnTab = sessionStorage.getItem(RETURN_TAB_KEY) || '';
	    sessionStorage.removeItem(RETURN_TAB_KEY);
	    if (returnTab && tabs.buttons.some(button => button.dataset.jobtrackTab === returnTab)) {
	      activateTab(returnTab);
	    }
	    return true;
	  };

	  const updateAuthUI = () => {
	    const authed = authIsValid(state.auth);
	    if (els.signIn) {
	      els.signIn.disabled = authed;
	      els.signIn.setAttribute('aria-disabled', authed ? 'true' : 'false');
	    }
	    if (els.signOut) {
	      els.signOut.disabled = !authed;
	      els.signOut.setAttribute('aria-disabled', !authed ? 'true' : 'false');
	    }
	    if (els.authStatus) {
	      if (authed) {
	        clearAuthMessage();
	        const claims = getAuthClaims(state.auth);
	        const label = claims.email || claims['cognito:username'] || claims.username || 'Signed in';
	        setStatus(els.authStatus, `Signed in as ${label}.`, 'success');
	      } else if (authUiMessage) {
	        setStatus(els.authStatus, authUiMessage, authUiTone);
	      } else {
	        setStatus(els.authStatus, 'Sign in to continue.', 'info');
	      }
	    }
	    tabs.buttons.forEach((button) => {
	      const disabled = !authed;
	      button.disabled = disabled;
	      button.setAttribute('aria-disabled', disabled ? 'true' : 'false');
	    });
	    els.jumpEntryButtons.forEach((button) => {
	      button.disabled = !authed;
	      button.setAttribute('aria-disabled', !authed ? 'true' : 'false');
	    });
	    els.jumpTabButtons.forEach((button) => {
	      button.disabled = !authed;
	      button.setAttribute('aria-disabled', !authed ? 'true' : 'false');
	    });
	    if (els.importStatus) {
	      setStatus(els.importStatus, authed ? 'Ready to import applications.' : 'Sign in to import applications.', authed ? '' : 'info');
	    }
	    if (els.prospectImportStatus) {
	      setStatus(els.prospectImportStatus, authed ? 'Ready to import prospects.' : 'Sign in to import prospects.', authed ? '' : 'info');
    }
    if (els.entryFormStatus && !state.editingEntryId) {
      const entryLabel = state.entryType === 'prospect' ? 'prospects' : 'applications';
      setStatus(els.entryFormStatus, authed ? `Ready to save ${entryLabel}.` : 'Sign in to save entries.', authed ? '' : 'info');
    }
    if (els.entryListStatus) {
      setStatus(els.entryListStatus, authed ? 'Use filters and sorting to review your entries.' : 'Sign in to load your entries.', authed ? '' : 'info');
    }
    if (els.entrySummary && !authed) {
      els.entrySummary.textContent = 'Sign in to view summary.';
    }
    if (els.prospectReviewStatus) {
      setStatus(els.prospectReviewStatus, authed ? 'Review your prospect queue.' : 'Sign in to load prospects.', authed ? '' : 'info');
    }
	    if (els.exportStatus) {
	      setStatus(els.exportStatus, authed ? 'Choose a date range to export applications.' : 'Sign in to export applications.', authed ? '' : 'info');
	    }
	    if (authed) {
	      closeAuthModal();
	    } else {
	      openAuthModal();
	    }
	    startAuthWatcher();
	  };

	  const getAuthHeader = () => {
	    if (!authIsValid(state.auth)) return null;
	    if (!state.auth || !state.auth.idToken) return null;
	    return `Bearer ${state.auth.idToken}`;
	  };

	  const requestJson = async (path, { method = 'GET', body } = {}) => {
	    if (!config.apiBase) throw new Error('API base URL is not configured.');
	    const authHeader = getAuthHeader();
	    if (!authHeader) {
	      clearAuth();
	      setAuthMessage('Sign in to continue.', 'info');
	      updateAuthUI();
	      throw new Error('Sign in to use the tracker.');
	    }
	    const res = await fetch(joinUrl(config.apiBase, path), {
	      method,
	      headers: {
	        'Content-Type': 'application/json',
	        Authorization: authHeader
	      },
	      body: body ? JSON.stringify(body) : undefined
	    });
	    const text = await res.text();
	    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
	    if (!res.ok) {
	      if (res.status === 401 || res.status === 403) {
	        clearAuth();
	        setAuthMessage('Your session ended. Please sign in again.', 'error');
	        updateAuthUI();
	      }
	      throw new Error(data?.error || data?.message || text || `${res.status} ${res.statusText}`);
	    }
	    return data || {};
	  };

  const collectAttachments = () => {
    const attachments = [];
    const resumeFile = els.resumeInput?.files?.[0];
    const coverFile = els.coverInput?.files?.[0];
    if (resumeFile) attachments.push({ file: resumeFile, kind: 'resume' });
    if (coverFile) attachments.push({ file: coverFile, kind: 'cover-letter' });
    return attachments;
  };

  const clearAttachmentInputs = () => {
    if (els.resumeInput) els.resumeInput.value = '';
    if (els.coverInput) els.coverInput.value = '';
  };

  const formatFileSize = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const validateAttachmentFiles = (attachments = [], statusEl) => {
    const maxBytes = config.maxAttachmentBytes || 0;
    if (!maxBytes) return true;
    for (const attachment of attachments) {
      const file = attachment?.file;
      if (!file || !Number.isFinite(file.size)) continue;
      if (file.size > maxBytes) {
        const limitLabel = formatFileSize(maxBytes);
        setStatus(statusEl, `${file.name || 'Attachment'} exceeds ${limitLabel}.`, 'error');
        return false;
      }
    }
    return true;
  };

  const updateAttachmentLimitText = () => {
    if (!els.attachmentLimit) return;
    const maxBytes = config.maxAttachmentBytes || 0;
    if (!maxBytes) return;
    els.attachmentLimit.textContent = `Files are stored privately with this application (max ${formatFileSize(maxBytes)} each).`;
  };

  const resetEntryDateFields = (type = state.entryType) => {
    setUnknownDateValue(els.postingDateInput, els.postingUnknownInput, '');
    if (type === 'application' && els.appliedDateInput) {
      els.appliedDateInput.value = els.appliedDateInput.value || formatDateInput(new Date());
    }
    if (type === 'prospect' && els.captureDateInput) {
      els.captureDateInput.value = els.captureDateInput.value || formatDateInput(new Date());
    }
  };

  const parseUrlSafe = (value = '') => {
    const raw = (value || '').toString().trim();
    if (!raw) return null;
    try {
      return new URL(raw);
    } catch {
      try {
        return new URL(`https://${raw}`);
      } catch {
        return null;
      }
    }
  };

  const isJobBoardHost = (host = '') => {
    const boardHosts = [
      'linkedin.com',
      'indeed.com',
      'glassdoor.com',
      'ziprecruiter.com',
      'monster.com',
      'careerbuilder.com',
      'simplyhired.com'
    ];
    return boardHosts.some(domain => host === domain || host.endsWith(`.${domain}`));
  };

  const cleanCompanyName = (value = '') => {
    const cleaned = value
      .replace(/[-_]+/g, ' ')
      .replace(/\bcareers?\b|\bjobs?\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    return cleaned ? toTitle(cleaned) : '';
  };

  const extractCompanyFromUrl = (value) => {
    const parsed = value instanceof URL ? value : parseUrlSafe(value);
    if (!parsed) return '';
    const host = parsed.hostname.toLowerCase().replace(/^www\./, '');
    if (isJobBoardHost(host)) return '';
    const pathParts = parsed.pathname.split('/').filter(Boolean);
    if (host.includes('greenhouse.io') || host.includes('lever.co') || host.includes('ashbyhq.com')) {
      return cleanCompanyName(pathParts[0] || '');
    }
    if (host.includes('workdayjobs.com') || host.includes('myworkdayjobs.com')) {
      return cleanCompanyName(pathParts[0] || host.split('.')[0]);
    }
    return cleanCompanyName(host.split('.')[0] || '');
  };

  const extractRoleFromUrl = (value) => {
    const parsed = value instanceof URL ? value : parseUrlSafe(value);
    if (!parsed) return '';
    const parts = parsed.pathname.split('/').filter(Boolean);
    if (!parts.length) return '';
    let slug = parts[parts.length - 1];
    slug = slug.replace(/\.(html|php|aspx)$/i, '');
    try {
      slug = decodeURIComponent(slug);
    } catch {
      slug = slug.replace(/%[0-9a-f]{2}/gi, ' ');
    }
    slug = slug.replace(/[-_]+/g, ' ');
    slug = slug.replace(/\b(jobs?|careers?|openings?|positions?|req|requisition|posting)\b/gi, '');
    slug = slug.replace(/\s+/g, ' ').trim();
    if (!slug || /^\d+$/.test(slug)) return '';
    return toTitle(slug);
  };

  const maybeAutofillFromJobUrl = () => {
    if (!els.jobUrlInput) return;
    if (state.editingEntryId) return;
    const raw = (els.jobUrlInput.value || '').toString().trim();
    if (!raw) return;
    const parsed = parseUrlSafe(raw);
    if (!parsed) return;
    if (els.companyInput && !els.companyInput.value.trim()) {
      const company = extractCompanyFromUrl(parsed);
      if (company) els.companyInput.value = company;
    }
    if (els.titleInput && !els.titleInput.value.trim()) {
      const title = extractRoleFromUrl(parsed);
      if (title) els.titleInput.value = title;
    }
  };

  const setDraftStatus = (message, tone = '') => {
    if (!els.entryDraftStatus) return;
    els.entryDraftStatus.textContent = message;
    if (tone) {
      els.entryDraftStatus.dataset.tone = tone;
    } else {
      delete els.entryDraftStatus.dataset.tone;
    }
  };

  const formatTagInput = (tags = []) => (Array.isArray(tags) ? tags.join(', ') : '');

  const parseTagInput = (value) => {
    const raw = (value || '').toString().trim();
    if (!raw) return [];
    const seen = new Set();
    return raw.split(/[;,]+/)
      .map(tag => tag.trim())
      .filter(Boolean)
      .filter((tag) => {
        const key = tag.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  };

  const clearCustomFields = () => {
    if (!els.customFieldList) return;
    els.customFieldList.innerHTML = '';
  };

  const addCustomFieldRow = (key = '', value = '') => {
    if (!els.customFieldList) return;
    const row = document.createElement('div');
    row.className = 'jobtrack-custom-field-row';

    const keyInput = document.createElement('input');
    keyInput.type = 'text';
    keyInput.className = 'jobtrack-input';
    keyInput.placeholder = 'Field name';
    keyInput.value = key;
    keyInput.dataset.jobtrackCustom = 'key';

    const valueInput = document.createElement('input');
    valueInput.type = 'text';
    valueInput.className = 'jobtrack-input';
    valueInput.placeholder = 'Value';
    valueInput.value = value;
    valueInput.dataset.jobtrackCustom = 'value';

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn-ghost';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => {
      row.remove();
      scheduleEntryDraftSave();
    });

    row.appendChild(keyInput);
    row.appendChild(valueInput);
    row.appendChild(removeBtn);
    els.customFieldList.appendChild(row);
  };

  const setCustomFieldsFromEntry = (fields = {}) => {
    clearCustomFields();
    const entries = fields && typeof fields === 'object' ? Object.entries(fields) : [];
    if (!entries.length) return;
    entries.forEach(([key, value]) => addCustomFieldRow(key, value));
  };

  const readCustomFields = () => {
    if (!els.customFieldList) return {};
    const rows = [...els.customFieldList.querySelectorAll('.jobtrack-custom-field-row')];
    return rows.reduce((acc, row) => {
      const key = row.querySelector('[data-jobtrack-custom="key"]')?.value || '';
      const value = row.querySelector('[data-jobtrack-custom="value"]')?.value || '';
      const trimmedKey = key.toString().trim();
      const trimmedValue = value.toString().trim();
      if (trimmedKey && trimmedValue) acc[trimmedKey] = trimmedValue;
      return acc;
    }, {});
  };

  const buildEntryDraft = () => ({
    entryType: state.entryType,
    company: els.companyInput?.value || '',
    title: els.titleInput?.value || '',
    jobUrl: els.jobUrlInput?.value || '',
    location: els.locationInput?.value || '',
    source: els.sourceInput?.value || '',
    batch: els.batchInput?.value || '',
    postingDate: els.postingDateInput?.value || '',
    postingDateUnknown: Boolean(els.postingUnknownInput?.checked),
    appliedDate: els.appliedDateInput?.value || '',
    captureDate: els.captureDateInput?.value || '',
    status: els.statusInput?.value || '',
    notes: els.notesInput?.value || '',
    tags: parseTagInput(els.tagsInput?.value || ''),
    followUpDate: els.followUpDateInput?.value || '',
    followUpNote: els.followUpNoteInput?.value || '',
    customFields: readCustomFields()
  });

  const hasEntryDraftValues = (draft) => {
    if (!draft) return false;
    const fields = [
      draft.company,
      draft.title,
      draft.jobUrl,
      draft.location,
      draft.source,
      draft.batch,
      draft.postingDate,
      draft.notes,
      draft.followUpDate,
      draft.followUpNote
    ];
    if (fields.some(value => (value || '').toString().trim())) return true;
    if (Array.isArray(draft.tags) && draft.tags.length) return true;
    if (draft.customFields && Object.keys(draft.customFields).length) return true;
    return false;
  };

  const saveEntryDraft = () => {
    if (!els.entryForm || state.editingEntryId) return;
    const draft = buildEntryDraft();
    if (!hasEntryDraftValues(draft)) {
      try {
        localStorage.removeItem(ENTRY_DRAFT_KEY);
      } catch {}
      setDraftStatus('');
      return;
    }
    try {
      localStorage.setItem(ENTRY_DRAFT_KEY, JSON.stringify(draft));
      setDraftStatus('Draft saved.');
    } catch {
      setDraftStatus('');
    }
  };

  const scheduleEntryDraftSave = () => {
    if (state.entryDraftTimer) window.clearTimeout(state.entryDraftTimer);
    state.entryDraftTimer = window.setTimeout(saveEntryDraft, 400);
  };

  const clearEntryDraft = () => {
    try {
      localStorage.removeItem(ENTRY_DRAFT_KEY);
    } catch {}
    setDraftStatus('');
  };

  const restoreEntryDraft = () => {
    if (!els.entryForm || state.editingEntryId) return;
    let draft = null;
    try {
      draft = JSON.parse(localStorage.getItem(ENTRY_DRAFT_KEY) || 'null');
    } catch {
      draft = null;
    }
    if (!hasEntryDraftValues(draft)) return;
    const hasValues = [
      els.companyInput?.value,
      els.titleInput?.value,
      els.jobUrlInput?.value,
      els.locationInput?.value,
      els.sourceInput?.value,
      els.batchInput?.value,
      els.notesInput?.value,
      els.tagsInput?.value,
      els.followUpDateInput?.value,
      els.followUpNoteInput?.value
    ].some(value => (value || '').toString().trim());
    if (hasValues || (els.customFieldList && els.customFieldList.childElementCount)) return;
    setEntryType(draft.entryType || 'application', { preserveStatus: false });
    if (els.companyInput && draft.company) els.companyInput.value = draft.company;
    if (els.titleInput && draft.title) els.titleInput.value = draft.title;
    if (els.jobUrlInput && draft.jobUrl) els.jobUrlInput.value = draft.jobUrl;
    if (els.locationInput && draft.location) els.locationInput.value = draft.location;
    if (els.sourceInput && draft.source) els.sourceInput.value = draft.source;
    if (els.batchInput && draft.batch) els.batchInput.value = draft.batch;
    if (els.appliedDateInput && draft.appliedDate) els.appliedDateInput.value = draft.appliedDate;
    if (els.captureDateInput && draft.captureDate) els.captureDateInput.value = draft.captureDate;
    if (els.statusInput && draft.status) els.statusInput.value = draft.status;
    if (els.notesInput && draft.notes) els.notesInput.value = draft.notes;
    if (els.tagsInput && Array.isArray(draft.tags)) els.tagsInput.value = formatTagInput(draft.tags);
    if (els.followUpDateInput && draft.followUpDate) els.followUpDateInput.value = draft.followUpDate;
    if (els.followUpNoteInput && draft.followUpNote) els.followUpNoteInput.value = draft.followUpNote;
    if (draft.customFields) setCustomFieldsFromEntry(draft.customFields);
    setUnknownDateValue(
      els.postingDateInput,
      els.postingUnknownInput,
      draft.postingDate || '',
      draft.postingDateUnknown !== undefined ? draft.postingDateUnknown : true
    );
    setDraftStatus('Draft restored.', 'info');
  };

  const uploadAttachment = async (applicationId, attachment) => {
    const file = attachment.file;
    const contentType = file.type || 'application/octet-stream';
    const presign = await requestJson('/api/attachments/presign', {
      method: 'POST',
      body: {
        applicationId,
        filename: file.name || 'attachment',
        contentType,
        size: file.size
      }
    });
    const res = await fetch(presign.uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': contentType },
      body: file
    });
    if (!res.ok) {
      throw new Error('Unable to upload attachment.');
    }
    return {
      key: presign.key,
      filename: file.name || 'attachment',
      contentType,
      kind: attachment.kind || '',
      uploadedAt: new Date().toISOString(),
      size: file.size
    };
  };

  const uploadAttachments = async (applicationId, attachments = [], onProgress, statusEl = null) => {
    if (!validateAttachmentFiles(attachments, statusEl)) {
      throw new Error('Attachment exceeds size limit.');
    }
    const uploaded = [];
    for (const attachment of attachments) {
      try {
        const item = await uploadAttachment(applicationId, attachment);
        uploaded.push(item);
        if (onProgress) onProgress({ ok: true, attachment, item });
      } catch (err) {
        if (onProgress) onProgress({ ok: false, attachment, error: err });
        throw err;
      }
    }
    return uploaded;
  };

  const requestAttachmentZip = async (applicationId) => requestJson('/api/attachments/zip', {
    method: 'POST',
    body: { applicationId }
  });

  const defaultRange = () => {
    const end = new Date();
    const start = new Date();
    start.setUTCDate(end.getUTCDate() - 89);
    return { start, end };
  };

  const readRange = () => {
    const start = parseDateInput(els.filterStart?.value) || defaultRange().start;
    const end = parseDateInput(els.filterEnd?.value) || defaultRange().end;
    if (start > end) return { start: end, end: start };
    return { start, end };
  };

  const formatRangeLabel = (start, end) => {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return `${start.toLocaleDateString('en-US', options)} - ${end.toLocaleDateString('en-US', options)}`;
  };

  const updateRangeInputs = (range) => {
    if (els.filterStart) els.filterStart.value = formatDateInput(range.start);
    if (els.filterEnd) els.filterEnd.value = formatDateInput(range.end);
  };

  const buildQuery = (range) => {
    const params = new URLSearchParams({
      start: formatDateInput(range.start),
      end: formatDateInput(range.end)
    });
    return params.toString();
  };

	  const updateKpis = (summary) => {
	    if (els.kpiTotal) els.kpiTotal.textContent = summary.totalApplications ?? 0;
	    if (els.kpiInterviews) els.kpiInterviews.textContent = summary.interviews ?? 0;
	    if (els.kpiOffers) els.kpiOffers.textContent = summary.offers ?? 0;
	    if (els.kpiRejections) els.kpiRejections.textContent = summary.rejections ?? 0;
	  };

	  const DASHBOARD_VIEW_COPY = {
	    applications: {
	      title: 'Application pulse',
	      subtitle: 'Filter by date range to see momentum, status mix, daily activity, and efficiency gains.'
	    },
	    prospects: {
	      title: 'Prospect pulse',
	      subtitle: 'See pending prospects, what needs action next, and jump into your queue.'
	    }
	  };

	  const updateDashboardCopy = (view) => {
	    const safeView = view === 'prospects' ? 'prospects' : 'applications';
	    const copy = DASHBOARD_VIEW_COPY[safeView] || DASHBOARD_VIEW_COPY.applications;
	    if (els.dashboardTitle) els.dashboardTitle.textContent = copy.title;
	    if (els.dashboardSubtitle) els.dashboardSubtitle.textContent = copy.subtitle;
	  };

	  const setProspectDashboardStatus = (message, tone = '') => {
	    if (!els.dashboardProspectStatus) return;
	    setStatus(els.dashboardProspectStatus, message, tone);
	  };

	  const resetProspectDashboard = () => {
	    if (els.kpiProspectsPending) els.kpiProspectsPending.textContent = '--';
	    if (els.kpiProspectsActive) els.kpiProspectsActive.textContent = '--';
	    if (els.kpiProspectsInterested) els.kpiProspectsInterested.textContent = '--';
	    if (els.kpiProspectsNeedsAction) els.kpiProspectsNeedsAction.textContent = '--';
	    if (els.prospectSummaryCount) els.prospectSummaryCount.textContent = '0 total';
	    if (els.prospectSummaryList) els.prospectSummaryList.innerHTML = '';
	  };

	  const isDateInRange = (date, range) => {
	    if (!date || !range) return false;
	    const dateKey = getDayKey(date);
	    const startKey = getDayKey(range.start);
	    const endKey = getDayKey(range.end);
	    return dateKey >= startKey && dateKey <= endKey;
	  };

	  const getProspectDashboardEntries = (range) => {
	    const entries = Array.isArray(state.entries) ? state.entries : [];
	    return entries.filter((entry) => {
	      if (!entry) return false;
	      const entryType = entry.entryType || getEntryType(entry);
	      if (entryType !== 'prospect') return false;
	      if (getEntryStatusGroup(entry) !== 'active') return false;
	      const captured = getEntryDate(entry);
	      if (!captured) return false;
	      return isDateInRange(captured, range);
	    });
	  };

	  const renderProspectDashboardList = (entries = []) => {
	    if (!els.prospectSummaryList) return;
	    els.prospectSummaryList.innerHTML = '';
	    if (!entries.length) {
	      const empty = document.createElement('li');
	      empty.className = 'jobtrack-prospect-empty';
	      empty.textContent = 'No pending prospects in this date range.';
	      els.prospectSummaryList.appendChild(empty);
	      return;
	    }
	    entries.forEach((entry) => {
	      const entryId = entry?.applicationId || '';
	      const item = document.createElement('li');
	      const button = document.createElement('button');
	      button.type = 'button';
	      button.className = 'jobtrack-detail-entry';
	      if (entryId) {
	        button.dataset.id = entryId;
	        button.setAttribute('aria-label', `View ${getEntryLabel(entry)}`);
	        button.addEventListener('click', () => openDetailModal(entry));
	      } else {
	        button.disabled = true;
	        button.setAttribute('aria-disabled', 'true');
	      }

	      const title = document.createElement('span');
	      title.className = 'jobtrack-detail-entry-title';
	      title.textContent = getEntryLabel(entry);
	      const meta = document.createElement('span');
	      meta.className = 'jobtrack-detail-entry-meta';
	      const action = getNextAction(entry);
	      meta.textContent = [action.label, formatProspectMeta(entry)].filter(Boolean).join(' · ') || 'View entry details';

	      button.appendChild(title);
	      button.appendChild(meta);
	      item.appendChild(button);
	      els.prospectSummaryList.appendChild(item);
	    });
	  };

	  const updateProspectDashboard = () => {
	    if (!els.dashboardProspects) return;
	    if (!config.apiBase) {
	      resetProspectDashboard();
	      setProspectDashboardStatus('Set the API base URL to load prospects.', 'error');
	      return;
	    }
		    if (!authIsValid(state.auth)) {
		      resetProspectDashboard();
		      setProspectDashboardStatus('Sign in to load prospects.', 'info');
		      return;
		    }
		    if (!state.entriesLoaded) {
		      resetProspectDashboard();
		      setProspectDashboardStatus('Loading prospects...', 'info');
		      return;
		    }
		    const range = state.range || readRange();
		    state.range = range;
		    updateRangeInputs(range);

	    const prospects = getProspectDashboardEntries(range);
	    const pendingCount = prospects.length;
	    let activeCount = 0;
	    let interestedCount = 0;
	    let needsActionCount = 0;

	    prospects.forEach((entry) => {
	      const statusKey = getEntryStatusKey(entry);
	      if (statusKey === 'active') activeCount += 1;
	      if (statusKey === 'interested') interestedCount += 1;
	      const actionTone = getNextAction(entry).tone;
	      if (actionTone === 'danger' || actionTone === 'warning') needsActionCount += 1;
	    });

	    if (els.kpiProspectsPending) els.kpiProspectsPending.textContent = pendingCount;
	    if (els.kpiProspectsActive) els.kpiProspectsActive.textContent = activeCount;
	    if (els.kpiProspectsInterested) els.kpiProspectsInterested.textContent = interestedCount;
	    if (els.kpiProspectsNeedsAction) els.kpiProspectsNeedsAction.textContent = needsActionCount;
	    if (els.prospectSummaryCount) {
	      els.prospectSummaryCount.textContent = `${pendingCount} total`;
	    }

	    const sorted = [...prospects].sort((a, b) => {
	      const toneRank = (entry) => {
	        const tone = getNextAction(entry).tone;
	        if (tone === 'danger') return 0;
	        if (tone === 'warning') return 1;
	        return 2;
	      };
	      const aRank = toneRank(a);
	      const bRank = toneRank(b);
	      if (aRank !== bRank) return aRank - bRank;
	      const aDate = getEntryDate(a);
	      const bDate = getEntryDate(b);
	      const aTime = aDate ? aDate.getTime() : 0;
	      const bTime = bDate ? bDate.getTime() : 0;
	      if (aTime !== bTime) return aTime - bTime;
	      return getEntryLabel(a).localeCompare(getEntryLabel(b), 'en', { sensitivity: 'base' });
	    }).slice(0, 8);

	    renderProspectDashboardList(sorted);
	    setProspectDashboardStatus(
	      pendingCount
	        ? `Loaded ${pendingCount} pending prospect${pendingCount === 1 ? '' : 's'}.`
	        : 'No pending prospects in this date range.',
	      'success'
	    );
	  };

	  const setDashboardView = (view) => {
	    const nextView = view === 'prospects' ? 'prospects' : 'applications';
	    state.dashboardView = nextView;
	    if (els.dashboardApplications) els.dashboardApplications.hidden = nextView !== 'applications';
	    if (els.dashboardProspects) els.dashboardProspects.hidden = nextView !== 'prospects';
	    if (els.dashboardViewInputs.length) {
	      els.dashboardViewInputs.forEach((input) => {
	        input.checked = input.value === nextView;
	      });
	    }
	    updateDashboardCopy(nextView);
	    if (nextView === 'applications') {
	      window.requestAnimationFrame(() => {
	        if (state.lineChart) state.lineChart.resize();
	        if (state.statusChart) state.statusChart.resize();
	      });
	    } else {
	      updateProspectDashboard();
	    }
	    try {
	      localStorage.setItem(DASHBOARD_VIEW_KEY, nextView);
	    } catch {}
	  };

	  const initDashboardView = () => {
	    if (!els.dashboardApplications || !els.dashboardProspects) return;
	    let stored = '';
	    try {
	      stored = localStorage.getItem(DASHBOARD_VIEW_KEY) || '';
	    } catch {}
	    setDashboardView(stored || state.dashboardView);
	    if (els.dashboardViewInputs.length) {
	      els.dashboardViewInputs.forEach((input) => {
	        input.addEventListener('change', () => setDashboardView(input.value));
	      });
	    }
	    if (els.dashboardProspectOpen) {
	      els.dashboardProspectOpen.addEventListener('click', () => {
	        activateTab('prospects', true);
	        window.requestAnimationFrame(() => {
	          if (els.prospectReviewList && typeof els.prospectReviewList.scrollIntoView === 'function') {
	            els.prospectReviewList.scrollIntoView({ behavior: 'smooth', block: 'start' });
	          }
	        });
	      });
	    }
	  };

	  const startOfWeek = (date) => {
	    const start = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
	    const day = start.getUTCDay();
    const diff = (day + 6) % 7;
    start.setUTCDate(start.getUTCDate() - diff);
    return start;
  };

  const formatWeekLabel = (start, end) => {
    const includeYear = start.getUTCFullYear() !== end.getUTCFullYear();
    if (start.getUTCMonth() === end.getUTCMonth() && start.getUTCFullYear() === end.getUTCFullYear()) {
      const month = start.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
      const yearSuffix = includeYear ? `, ${start.getUTCFullYear()}` : '';
      return `${month} ${start.getUTCDate()}–${end.getUTCDate()}${yearSuffix}`;
    }
    const options = includeYear
      ? { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }
      : { month: 'short', day: 'numeric', timeZone: 'UTC' };
    const startLabel = start.toLocaleDateString('en-US', options);
    const endLabel = end.toLocaleDateString('en-US', options);
    return `${startLabel}–${endLabel}`;
  };

  const groupSeriesByWeek = (series = []) => {
    const buckets = new Map();
    series.forEach((item) => {
      const date = parseDateInput(item?.date);
      if (!date) return;
      const weekStart = startOfWeek(date);
      const key = formatDateInput(weekStart);
      const bucket = buckets.get(key) || {
        start: weekStart,
        minDate: date,
        maxDate: date,
        count: 0
      };
      bucket.count += item?.count || 0;
      if (date < bucket.minDate) bucket.minDate = date;
      if (date > bucket.maxDate) bucket.maxDate = date;
      buckets.set(key, bucket);
    });
    return Array.from(buckets.values())
      .sort((a, b) => a.start - b.start)
      .map(bucket => ({
        key: formatDateInput(bucket.start),
        label: formatWeekLabel(bucket.minDate, bucket.maxDate),
        count: bucket.count,
        start: bucket.minDate,
        end: bucket.maxDate
      }));
  };

  const updateLineChart = (series, rangeLabel) => {
    if (els.lineRange) els.lineRange.textContent = rangeLabel;
    if (els.lineTotal) {
      const total = series.reduce((acc, item) => acc + (item.count || 0), 0);
      els.lineTotal.textContent = `${total} apps`;
    }
    const ctx = document.getElementById('jobtrack-line-chart');
    if (!ctx || !window.Chart) return;
    const weeklySeries = groupSeriesByWeek(series);
    state.weeklySeries = weeklySeries;
    const labels = weeklySeries.map(item => item.label);
    const data = weeklySeries.map(item => item.count);
    const accent = readCssVar('--jobtrack-accent', '#2396AD');
    const grid = toRgba('#ffffff', 0.08);
    const text = readCssVar('--text-muted', '#BFC8D3');
    const dataset = {
      label: 'Applications',
      data,
      borderColor: accent,
      backgroundColor: toRgba(accent, 0.2),
      tension: 0.35,
      fill: true,
      pointRadius: 2,
      pointHoverRadius: 4,
      borderWidth: 2
    };

    const handleClick = (event, elements) => {
      if (!elements || !elements.length) return;
      const idx = elements[0].index;
      const item = state.weeklySeries[idx];
      if (!item) return;
      showWeekDetail(item.key, item.label, item.count);
    };
    if (state.lineChart) {
      state.lineChart.data.labels = labels;
      state.lineChart.data.datasets = [dataset];
      state.lineChart.options.onClick = handleClick;
      state.lineChart.update();
      return;
    }
    state.lineChart = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [dataset]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        onClick: handleClick,
        scales: {
          x: {
            ticks: { color: text, maxTicksLimit: 6 },
            grid: { color: grid }
          },
          y: {
            ticks: { color: text, precision: 0 },
            grid: { color: grid },
            beginAtZero: true
          }
        }
      }
    });
  };

  const updateStatusChart = (statuses) => {
    if (els.statusTotal) {
      const total = statuses.reduce((acc, item) => acc + (item.count || 0), 0);
      els.statusTotal.textContent = `${total} statuses`;
    }
    const ctx = document.getElementById('jobtrack-status-chart');
    if (!ctx || !window.Chart) return;
    const labels = statuses.map(item => item.status);
    const data = statuses.map(item => item.count);
    const accent = readCssVar('--jobtrack-accent', '#2396AD');
    const grid = toRgba('#ffffff', 0.08);
    const text = readCssVar('--text-muted', '#BFC8D3');
    const dataset = {
      label: 'Applications',
      data,
      backgroundColor: toRgba(accent, 0.35),
      borderColor: accent,
      borderWidth: 1.5
    };
    const handleClick = (event, elements) => {
      if (!elements || !elements.length) return;
      const idx = elements[0].index;
      const status = labels[idx];
      if (!status) return;
      showStatusDetail(status);
    };
    if (state.statusChart) {
      state.statusChart.data.labels = labels;
      state.statusChart.data.datasets = [dataset];
      state.statusChart.options.onClick = handleClick;
      state.statusChart.update();
      return;
    }
    state.statusChart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [dataset]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        onClick: handleClick,
        scales: {
          x: {
            ticks: { color: text, precision: 0 },
            grid: { color: grid },
            beginAtZero: true
          },
          y: {
            ticks: { color: text },
            grid: { color: 'transparent' }
          }
        }
      }
    });
  };

  const buildCalendar = (days, rangeLabel, range, detailMap = null) => {
    if (!els.calendarGrid) return;
    if (els.calendarRange) els.calendarRange.textContent = rangeLabel;
    els.calendarGrid.innerHTML = '';
    if (els.calendarMonths) els.calendarMonths.innerHTML = '';
    const counts = new Map(days.map(item => [item.date, item.count]));
    const max = Math.max(0, ...days.map(item => item.count || 0));
    const scale = (count) => {
      if (!count) return 0;
      if (max <= 1) return 1;
      return Math.min(4, Math.ceil((count / max) * 4));
    };

    const startMonth = new Date(Date.UTC(range.start.getUTCFullYear(), range.start.getUTCMonth(), 1));
    const endMonth = new Date(Date.UTC(range.end.getUTCFullYear(), range.end.getUTCMonth(), 1));
    const cursor = new Date(startMonth);

    while (cursor <= endMonth) {
      const year = cursor.getUTCFullYear();
      const month = cursor.getUTCMonth();
      const monthLabel = new Date(Date.UTC(year, month, 1)).toLocaleDateString('en-US', {
        month: 'short',
        timeZone: 'UTC'
      });

      if (els.calendarMonths) {
        const label = document.createElement('span');
        label.className = 'jobtrack-calendar-month';
        label.textContent = monthLabel;
        els.calendarMonths.appendChild(label);
      }

      const block = document.createElement('div');
      block.className = 'jobtrack-calendar-month-block';
      const grid = document.createElement('div');
      grid.className = 'jobtrack-calendar-month-grid';

      const firstDay = new Date(Date.UTC(year, month, 1));
      const firstOffset = firstDay.getUTCDay();
      const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
      const totalSlots = 42;

      for (let slot = 0; slot < totalSlots; slot += 1) {
        const dayNumber = slot - firstOffset + 1;
        const cell = document.createElement('div');
        cell.className = 'jobtrack-calendar-day';
        if (dayNumber < 1 || dayNumber > daysInMonth) {
          cell.dataset.empty = 'true';
          cell.setAttribute('aria-hidden', 'true');
          grid.appendChild(cell);
          continue;
        }
        const date = new Date(Date.UTC(year, month, dayNumber));
        const iso = formatDateInput(date);
        const count = counts.get(iso) || 0;
        const intensity = scale(count);
        if (intensity) {
          cell.dataset.intensity = String(intensity);
        } else {
          cell.removeAttribute('data-intensity');
        }
        const dateLabel = date.toLocaleDateString('en-US', {
          weekday: 'short',
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          timeZone: 'UTC'
        });
        const labelText = count
          ? `${dateLabel}: ${count} application${count === 1 ? '' : 's'}`
          : `${dateLabel}: No applications`;
        const detail = detailMap?.get(iso);
        const detailLines = count && detail ? formatDetailLines(detail) : [];
        const tooltipText = detailLines.length ? [labelText, ...detailLines].join('\n') : labelText;
        cell.setAttribute('role', 'gridcell');
        cell.setAttribute('aria-label', tooltipText);
        cell.title = tooltipText;
        cell.dataset.jobtrackTooltip = tooltipText;
        cell.dataset.jobtrackDate = iso;
        cell.tabIndex = 0;
        grid.appendChild(cell);
      }

      block.appendChild(grid);
      els.calendarGrid.appendChild(block);
      cursor.setUTCMonth(month + 1);
    }
    bindCalendarTooltip();
  };

  const buildWeekdayHeatmap = (weekdayDetails = new Map()) => {
    if (!els.weekdayHeatmap) return;
    els.weekdayHeatmap.innerHTML = '';
    const counts = WEEKDAYS.map(day => weekdayDetails.get(day)?.total || 0);
    const max = Math.max(0, ...counts);
    const scale = (count) => {
      if (!count) return 0;
      if (max <= 1) return 1;
      return Math.min(4, Math.ceil((count / max) * 4));
    };
    WEEKDAYS.forEach((day) => {
      const detail = weekdayDetails.get(day);
      const count = detail?.total || 0;
      const intensity = scale(count);
      const cell = document.createElement('div');
      cell.className = 'jobtrack-weekday-cell';
      if (intensity) {
        cell.dataset.intensity = String(intensity);
      }
      const labelText = `${day}: ${count} application${count === 1 ? '' : 's'}`;
      const detailLines = count && detail ? formatDetailLines(detail) : [];
      const tooltipText = detailLines.length ? [labelText, ...detailLines].join('\n') : labelText;
      cell.setAttribute('role', 'gridcell');
      cell.setAttribute('aria-label', tooltipText);
      cell.title = tooltipText;
      cell.dataset.jobtrackTooltip = tooltipText;
      cell.dataset.jobtrackWeekday = day;
      cell.tabIndex = 0;
      els.weekdayHeatmap.appendChild(cell);
    });
    bindWeekdayTooltip();
  };

  const getFirstResponseDate = (entry) => {
    const history = Array.isArray(entry?.statusHistory) ? entry.statusHistory : [];
    if (!history.length) {
      const status = (entry?.status || '').toString().trim().toLowerCase();
      if (status && status !== 'applied') {
        return parseIsoDate(entry?.updatedAt) || parseIsoDate(entry?.createdAt);
      }
      return null;
    }
    const sorted = history
      .map(item => ({
        status: (item?.status || '').toString().trim().toLowerCase(),
        date: parseIsoDate(item?.date) || parseDateInput(item?.date)
      }))
      .filter(item => item.date)
      .sort((a, b) => a.date - b.date);
    for (const item of sorted) {
      if (item.status && item.status !== 'applied') return item.date;
    }
    return null;
  };

  const buildInsights = (items = []) => {
    let foundSum = 0;
    let foundCount = 0;
    let postedSum = 0;
    let postedCount = 0;
    let responseSum = 0;
    let responseCount = 0;
    const sourceCounts = new Map();
    const weekdayCounts = new Map(WEEKDAYS.map(day => [day, 0]));

    items.forEach((item) => {
      const applied = parseDateInput(item?.appliedDate);
      if (!applied) return;
      const appliedTime = applied.getTime();

      const captureDate = parseDateInput(item?.captureDate);
      if (captureDate) {
        const delta = (appliedTime - captureDate.getTime()) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          foundSum += delta;
          foundCount += 1;
        }
      }

      const postingDate = parseDateInput(item?.postingDate);
      if (postingDate) {
        const delta = (appliedTime - postingDate.getTime()) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          postedSum += delta;
          postedCount += 1;
        }
      }

      const responseDate = getFirstResponseDate(item);
      if (responseDate) {
        const delta = (responseDate.getTime() - appliedTime) / 86400000;
        if (Number.isFinite(delta) && delta >= 0) {
          responseSum += delta;
          responseCount += 1;
        }
      }

      const source = (item?.source || '').toString().trim();
      if (source) {
        const key = source.toLowerCase();
        const current = sourceCounts.get(key) || { label: source, count: 0 };
        current.count += 1;
        sourceCounts.set(key, current);
      }

      const weekday = WEEKDAYS[applied.getUTCDay()];
      if (weekday) {
        weekdayCounts.set(weekday, (weekdayCounts.get(weekday) || 0) + 1);
      }
    });

    let topSource = null;
    let topSourceCount = 0;
    sourceCounts.forEach((value) => {
      if (value.count > topSourceCount) {
        topSourceCount = value.count;
        topSource = value.label;
      }
    });

    let bestWeekday = null;
    let bestWeekdayCount = 0;
    weekdayCounts.forEach((count, day) => {
      if (count > bestWeekdayCount) {
        bestWeekdayCount = count;
        bestWeekday = day;
      }
    });

    const total = items.length;
    return {
      avgFoundToApplied: foundCount ? foundSum / foundCount : null,
      foundCount,
      avgPostedToApplied: postedCount ? postedSum / postedCount : null,
      postedCount,
      avgResponseTime: responseCount ? responseSum / responseCount : null,
      responseCount,
      responseRate: total ? (responseCount / total) * 100 : null,
      total,
      topSource: topSource || null,
      topSourceCount,
      bestWeekday: bestWeekday || null,
      bestWeekdayCount
    };
  };

  const createDetail = () => ({
    total: 0,
    statuses: new Map(),
    companies: new Map(),
    start: null,
    end: null
  });

  const updateDetail = (detail, status, company, date) => {
    detail.total += 1;
    if (status) incrementCount(detail.statuses, status);
    if (company) incrementCount(detail.companies, company);
    if (date) {
      if (!detail.start || date < detail.start) detail.start = date;
      if (!detail.end || date > detail.end) detail.end = date;
    }
  };

  const buildDashboardDetails = (applications = []) => {
    const mapDetails = new Map();
    const remoteDetail = createDetail();
    const calendarDetails = new Map();
    const statusDetails = new Map();
    const weekDetails = new Map();
    const weekdayDetails = new Map();

    applications.forEach((item) => {
      const status = toTitle((item?.status || 'Applied').toString());
      const company = (item?.company || '').toString().trim();
      const appliedDate = item?.appliedDate;
      const applied = parseDateInput(appliedDate);

      if (applied) {
        const dateKey = formatDateInput(applied);
        if (!calendarDetails.has(dateKey)) calendarDetails.set(dateKey, createDetail());
        updateDetail(calendarDetails.get(dateKey), status, company, applied);

        const weekStart = startOfWeek(applied);
        const weekKey = formatDateInput(weekStart);
        if (!weekDetails.has(weekKey)) weekDetails.set(weekKey, createDetail());
        updateDetail(weekDetails.get(weekKey), status, company, applied);

        const weekday = WEEKDAYS[applied.getUTCDay()];
        if (weekday) {
          if (!weekdayDetails.has(weekday)) weekdayDetails.set(weekday, createDetail());
          updateDetail(weekdayDetails.get(weekday), status, company, applied);
        }
      }

      if (status) {
        if (!statusDetails.has(status)) statusDetails.set(status, createDetail());
        updateDetail(statusDetails.get(status), status, company, applied);
      }

      const location = (item?.location || '').toString();
      if (!location) return;
      if (isRemoteLocation(location)) {
        updateDetail(remoteDetail, status, company, applied);
        return;
      }
      const code = extractStateCode(location);
      if (!code) return;
      if (!mapDetails.has(code)) mapDetails.set(code, createDetail());
      updateDetail(mapDetails.get(code), status, company, applied);
    });

    return {
      mapDetails,
      remoteDetail,
      calendarDetails,
      statusDetails,
      weekDetails,
      weekdayDetails
    };
  };

  const formatDetailLines = (detail) => {
    if (!detail || !detail.total) {
      return ['No applications recorded in this view yet.'];
    }
    const lines = [
      `Statuses: ${formatCountList(detail.statuses, 5)}`,
      `Top companies: ${formatCountList(detail.companies, 4)}`
    ];
    if (detail.start) {
      const startLabel = formatDateLabel(formatDateInput(detail.start));
      const endLabel = detail.end ? formatDateLabel(formatDateInput(detail.end)) : startLabel;
      const rangeLabel = startLabel === endLabel ? startLabel : `${startLabel}–${endLabel}`;
      lines.push(`Activity span: ${rangeLabel}`);
    }
    return lines;
  };

  const getDashboardEntries = () => (Array.isArray(state.dashboardEntries) ? state.dashboardEntries : []);

  const showMapDetail = (code, label, kind = 'on-site') => {
    if (!code) return;
    const detail = code === 'REMOTE'
      ? state.mapDetails?.remote
      : state.mapDetails?.states?.get(code);
    const total = detail?.total || 0;
    const title = `${label}: ${total} ${kind} application${total === 1 ? '' : 's'}`;
    const entries = getDashboardEntries().filter((entry) => {
      const location = (entry?.location || '').toString();
      if (!location) return false;
      if (code === 'REMOTE') return isRemoteLocation(location);
      if (isRemoteLocation(location)) return false;
      return extractStateCode(location) === code;
    });
    setDashboardDetail(title, formatDetailLines(detail), entries);
  };

  const showCalendarDetail = (dateKey) => {
    if (!dateKey) return;
    const detail = state.calendarDetails?.get(dateKey);
    const total = detail?.total || 0;
    const title = `${formatDateLabel(dateKey)}: ${total} application${total === 1 ? '' : 's'}`;
    const entries = getDashboardEntries().filter((entry) => {
      const date = parseDateInput(entry?.appliedDate);
      return date ? formatDateInput(date) === dateKey : false;
    });
    setDashboardDetail(title, formatDetailLines(detail), entries);
  };

  const showStatusDetail = (status) => {
    if (!status) return;
    const detail = state.statusDetails?.get(status);
    const total = detail?.total || 0;
    const title = `${status}: ${total} application${total === 1 ? '' : 's'}`;
    const target = status.toString().toLowerCase();
    const entries = getDashboardEntries().filter((entry) => getEntryStatusLabel(entry).toLowerCase() === target);
    setDashboardDetail(title, formatDetailLines(detail), entries);
  };

  const showWeekDetail = (weekKey, label, count) => {
    if (!weekKey) return;
    const detail = state.weekDetails?.get(weekKey);
    const total = detail?.total ?? count ?? 0;
    const title = `Week of ${label}: ${total} application${total === 1 ? '' : 's'}`;
    const start = parseDateInput(weekKey);
    const end = start ? new Date(start) : null;
    if (end) end.setUTCDate(end.getUTCDate() + 6);
    const entries = start && end
      ? getDashboardEntries().filter((entry) => {
        const date = parseDateInput(entry?.appliedDate);
        return date ? date >= start && date <= end : false;
      })
      : [];
    setDashboardDetail(title, formatDetailLines(detail), entries);
  };

  const showWeekdayDetail = (weekday) => {
    if (!weekday) return;
    const detail = state.weekdayDetails?.get(weekday);
    const total = detail?.total || 0;
    const title = `${weekday}: ${total} application${total === 1 ? '' : 's'}`;
    const entries = getDashboardEntries().filter((entry) => {
      const date = parseDateInput(entry?.appliedDate);
      return date ? WEEKDAYS[date.getUTCDay()] === weekday : false;
    });
    setDashboardDetail(title, formatDetailLines(detail), entries);
  };

  const updateInsights = (insights = {}) => {
    if (els.kpiFoundToApplied) {
      els.kpiFoundToApplied.textContent = formatDays(insights.avgFoundToApplied);
    }
    if (els.kpiFoundCount) {
      els.kpiFoundCount.textContent = insights.foundCount
        ? `${insights.foundCount} apps`
        : 'No data yet';
    }
    if (els.kpiPostedToApplied) {
      els.kpiPostedToApplied.textContent = formatDays(insights.avgPostedToApplied);
    }
    if (els.kpiPostedCount) {
      els.kpiPostedCount.textContent = insights.postedCount
        ? `${insights.postedCount} apps`
        : 'No data yet';
    }
    if (els.kpiResponseTime) {
      els.kpiResponseTime.textContent = formatDays(insights.avgResponseTime);
    }
    if (els.kpiResponseTimeCount) {
      els.kpiResponseTimeCount.textContent = insights.responseCount
        ? `${insights.responseCount} responses`
        : 'No responses yet';
    }
    if (els.kpiResponseRate) {
      els.kpiResponseRate.textContent = formatPercent(insights.responseRate);
    }
    if (els.kpiResponseCount) {
      els.kpiResponseCount.textContent = insights.total
        ? `${insights.responseCount} of ${insights.total} apps`
        : 'No data yet';
    }
    if (els.kpiTopSource) {
      els.kpiTopSource.textContent = insights.topSource || '--';
    }
    if (els.kpiTopSourceCount) {
      els.kpiTopSourceCount.textContent = insights.topSourceCount
        ? `${insights.topSourceCount} apps`
        : 'No sources yet';
    }
    if (els.kpiBestWeekday) {
      els.kpiBestWeekday.textContent = insights.bestWeekday || '--';
    }
    if (els.kpiBestWeekdayCount) {
      els.kpiBestWeekdayCount.textContent = insights.bestWeekdayCount
        ? `${insights.bestWeekdayCount} apps`
        : 'No data yet';
    }
  };

  const renderFunnel = (data = {}) => {
    if (!els.funnelList) return;
    const stages = Array.isArray(data.stages) ? data.stages : [];
    els.funnelList.innerHTML = '';
    if (!stages.length) {
      const empty = document.createElement('p');
      empty.className = 'jobtrack-form-status';
      empty.textContent = 'No funnel data yet.';
      els.funnelList.appendChild(empty);
      return;
    }
    const max = Math.max(0, ...stages.map(stage => stage.count || 0));
    stages.forEach((stage) => {
      const row = document.createElement('div');
      row.className = 'jobtrack-funnel-row';

      const meta = document.createElement('div');
      meta.className = 'jobtrack-funnel-meta';
      const label = document.createElement('span');
      label.textContent = stage.stage || 'Stage';
      const count = document.createElement('span');
      count.textContent = `${stage.count || 0} · ${formatPercent(stage.rate)}`;
      meta.appendChild(label);
      meta.appendChild(count);

      const bar = document.createElement('div');
      bar.className = 'jobtrack-funnel-bar';
      const fill = document.createElement('span');
      const width = max ? Math.round(((stage.count || 0) / max) * 100) : 0;
      fill.style.width = `${width}%`;
      bar.appendChild(fill);

      row.appendChild(meta);
      row.appendChild(bar);
      els.funnelList.appendChild(row);
    });
    const conversions = Array.isArray(data.conversions) ? data.conversions : [];
    if (conversions.length) {
      const conversionWrap = document.createElement('div');
      conversionWrap.className = 'jobtrack-funnel-conversions';
      conversions.forEach((conversion) => {
        const line = document.createElement('div');
        line.className = 'jobtrack-funnel-conversion';
        const label = `${conversion.from} → ${conversion.to}`;
        line.textContent = `${label}: ${formatPercent(conversion.rate)}`;
        conversionWrap.appendChild(line);
      });
      els.funnelList.appendChild(conversionWrap);
    }
  };

  const renderTimeInStage = (data = {}) => {
    if (!els.timeInStageList) return;
    const stages = Array.isArray(data.stages) ? data.stages : [];
    els.timeInStageList.innerHTML = '';
    if (!stages.length) {
      const empty = document.createElement('p');
      empty.className = 'jobtrack-form-status';
      empty.textContent = 'No timing data yet.';
      els.timeInStageList.appendChild(empty);
      return;
    }
    stages.forEach((stage) => {
      const row = document.createElement('div');
      row.className = 'jobtrack-time-row';
      const name = document.createElement('strong');
      name.textContent = stage.stage || 'Stage';
      const avg = document.createElement('span');
      avg.textContent = Number.isFinite(stage.avgDays) ? `Avg ${formatDays(stage.avgDays)}` : 'Avg --';
      const median = document.createElement('span');
      median.textContent = Number.isFinite(stage.medianDays) ? `Median ${formatDays(stage.medianDays)}` : 'Median --';
      row.appendChild(name);
      row.appendChild(avg);
      row.appendChild(median);
      els.timeInStageList.appendChild(row);
    });
  };

  const getMapStateCode = (node) => {
    const className = (node.getAttribute('class') || '').toString();
    const classes = className.split(/\s+/).filter(Boolean);
    const match = classes.find(value => STATE_CODE_SET.has(value.toLowerCase()));
    return match ? match.toUpperCase() : null;
  };

  const bindMapTooltip = (svg) => {
    if (!svg || state.mapTooltipBound) return;
    const handleMove = (event) => {
      const node = event.target.closest('path, circle');
      if (!node || !svg.contains(node)) {
        tooltip.hide();
        return;
      }
      const label = node.dataset.jobtrackTooltip;
      if (!label) {
        tooltip.hide();
        return;
      }
      tooltip.show(label, event.clientX, event.clientY);
    };
    const handleLeave = () => tooltip.hide();
    const handleFocus = (event) => {
      const node = event.target.closest('path, circle');
      if (!node || !svg.contains(node)) return;
      const label = node.dataset.jobtrackTooltip;
      if (!label) return;
      const rect = node.getBoundingClientRect();
      tooltip.show(label, rect.left + rect.width / 2, rect.top + rect.height / 2);
    };
    const handleClick = (event) => {
      const node = event.target.closest('path, circle');
      if (!node || !svg.contains(node)) return;
      const code = node.dataset.jobtrackState;
      const name = node.dataset.jobtrackStateName || STATE_NAME_LOOKUP.get(code) || code;
      if (!code) return;
      showMapDetail(code, name, 'on-site');
    };
    const handleKey = (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const node = event.target.closest('path, circle');
      if (!node || !svg.contains(node)) return;
      event.preventDefault();
      const code = node.dataset.jobtrackState;
      const name = node.dataset.jobtrackStateName || STATE_NAME_LOOKUP.get(code) || code;
      if (!code) return;
      showMapDetail(code, name, 'on-site');
    };
    svg.addEventListener('pointermove', handleMove);
    svg.addEventListener('pointerleave', handleLeave);
    svg.addEventListener('focusin', handleFocus);
    svg.addEventListener('focusout', handleLeave);
    svg.addEventListener('click', handleClick);
    svg.addEventListener('keydown', handleKey);
    state.mapTooltipBound = true;
  };

  const bindCalendarTooltip = () => {
    if (!els.calendarGrid || state.calendarTooltipBound) return;
    const handleMove = (event) => {
      const cell = event.target.closest('.jobtrack-calendar-day');
      if (!cell || !els.calendarGrid.contains(cell)) {
        tooltip.hide();
        return;
      }
      const label = cell.dataset.jobtrackTooltip;
      if (!label) {
        tooltip.hide();
        return;
      }
      tooltip.show(label, event.clientX, event.clientY);
    };
    const handleLeave = () => tooltip.hide();
    const handleClick = (event) => {
      const cell = event.target.closest('.jobtrack-calendar-day');
      if (!cell || !els.calendarGrid.contains(cell)) return;
      const dateKey = cell.dataset.jobtrackDate;
      showCalendarDetail(dateKey);
    };
    const handleKey = (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const cell = event.target.closest('.jobtrack-calendar-day');
      if (!cell || !els.calendarGrid.contains(cell)) return;
      event.preventDefault();
      const dateKey = cell.dataset.jobtrackDate;
      showCalendarDetail(dateKey);
    };
    els.calendarGrid.addEventListener('pointermove', handleMove);
    els.calendarGrid.addEventListener('pointerleave', handleLeave);
    els.calendarGrid.addEventListener('click', handleClick);
    els.calendarGrid.addEventListener('keydown', handleKey);
    state.calendarTooltipBound = true;
  };

  const bindWeekdayTooltip = () => {
    if (!els.weekdayHeatmap || state.weekdayTooltipBound) return;
    const handleMove = (event) => {
      const cell = event.target.closest('.jobtrack-weekday-cell');
      if (!cell || !els.weekdayHeatmap.contains(cell)) {
        tooltip.hide();
        return;
      }
      const label = cell.dataset.jobtrackTooltip;
      if (!label) {
        tooltip.hide();
        return;
      }
      tooltip.show(label, event.clientX, event.clientY);
    };
    const handleLeave = () => tooltip.hide();
    const handleClick = (event) => {
      const cell = event.target.closest('.jobtrack-weekday-cell');
      if (!cell || !els.weekdayHeatmap.contains(cell)) return;
      const weekday = cell.dataset.jobtrackWeekday;
      showWeekdayDetail(weekday);
    };
    const handleKey = (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const cell = event.target.closest('.jobtrack-weekday-cell');
      if (!cell || !els.weekdayHeatmap.contains(cell)) return;
      event.preventDefault();
      const weekday = cell.dataset.jobtrackWeekday;
      showWeekdayDetail(weekday);
    };
    els.weekdayHeatmap.addEventListener('pointermove', handleMove);
    els.weekdayHeatmap.addEventListener('pointerleave', handleLeave);
    els.weekdayHeatmap.addEventListener('click', handleClick);
    els.weekdayHeatmap.addEventListener('keydown', handleKey);
    state.weekdayTooltipBound = true;
  };

  const loadMap = async () => {
    if (!els.mapContainer || state.mapLoaded) return state.mapSvg;
    const src = (els.mapContainer.dataset.jobtrackMapSrc || '').trim();
    if (!src) {
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Map source missing.';
      return null;
    }
    try {
      const res = await fetch(src);
      if (!res.ok) throw new Error('Unable to load map.');
      const text = await res.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(text, 'image/svg+xml');
      const svg = doc.querySelector('svg');
      if (!svg) throw new Error('Map SVG not found.');
      svg.classList.add('jobtrack-map-svg');
      if (!svg.getAttribute('viewBox')) {
        const width = parseFloat(svg.getAttribute('width') || '0');
        const height = parseFloat(svg.getAttribute('height') || '0');
        if (width && height) {
          svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        }
      }
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.removeAttribute('width');
      svg.removeAttribute('height');
      const style = svg.querySelector('style');
      if (style) style.remove();
      els.mapContainer.innerHTML = '';
      els.mapContainer.appendChild(svg);
      state.mapLoaded = true;
      state.mapSvg = svg;
      return svg;
    } catch (err) {
      console.error('Map load failed', err);
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Unable to load map.';
      return null;
    }
  };

  const updateMap = async (applications = [], detailData = null) => {
    if (!els.mapContainer) return;
    const svg = await loadMap();
    if (!svg) return;

    const details = detailData || buildDashboardDetails(applications);
    const counts = new Map();
    let totalApps = 0;
    details.mapDetails.forEach((detail, code) => {
      counts.set(code, detail.total);
      totalApps += detail.total;
    });
    const remoteCount = details.remoteDetail?.total || 0;
    state.mapDetails = { states: details.mapDetails, remote: details.remoteDetail };

    const max = Math.max(0, ...Array.from(counts.values()));
    const scale = (count) => {
      if (!count) return 0;
      if (max <= 1) return 1;
      return Math.min(4, Math.ceil((count / max) * 4));
    };

    const nodes = svg.querySelectorAll('.state path, .state circle');
    nodes.forEach((node) => {
      const code = getMapStateCode(node);
      if (!code) return;
      const count = counts.get(code) || 0;
      const intensity = scale(count);
      if (intensity) {
        node.dataset.intensity = String(intensity);
      } else {
        node.removeAttribute('data-intensity');
      }
      const name = STATE_NAME_LOOKUP.get(code) || code;
      const share = totalApps ? Math.round((count / totalApps) * 100) : 0;
      const label = `${name}: ${count} on-site application${count === 1 ? '' : 's'}${share ? ` · ${share}% of on-site` : ''}`;
      const detail = details.mapDetails.get(code);
      const detailLines = detail?.total ? formatDetailLines(detail) : [];
      const tooltipText = detailLines.length ? [label, ...detailLines].join('\n') : label;
      node.dataset.jobtrackTooltip = tooltipText;
      node.dataset.jobtrackState = code;
      node.dataset.jobtrackStateName = name;
      node.setAttribute('aria-label', tooltipText);
      node.setAttribute('title', tooltipText);
      let titleNode = node.querySelector('title');
      if (!titleNode) {
        titleNode = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        node.appendChild(titleNode);
      }
      titleNode.textContent = tooltipText;
      node.setAttribute('tabindex', '0');
    });

    bindMapTooltip(svg);
    if (els.mapTotal) {
      const stateCount = counts.size;
      if (stateCount) {
        els.mapTotal.textContent = `${stateCount} state${stateCount === 1 ? '' : 's'} · ${totalApps} on-site · ${remoteCount} remote`;
      } else {
        els.mapTotal.textContent = remoteCount
          ? `No states yet · ${remoteCount} remote`
          : 'No states yet';
      }
    }
    if (els.mapRemote) {
      const label = `Remote: ${remoteCount} application${remoteCount === 1 ? '' : 's'}`;
      const detailLines = details.remoteDetail?.total ? formatDetailLines(details.remoteDetail) : [];
      const tooltipText = detailLines.length ? [label, ...detailLines].join('\n') : label;
      els.mapRemote.textContent = label;
      els.mapRemote.title = tooltipText;
      els.mapRemote.dataset.jobtrackTooltip = tooltipText;
      els.mapRemote.setAttribute('role', 'button');
      els.mapRemote.tabIndex = 0;
    }
  };

  const refreshDashboard = async () => {
    if (!els.dashboard || !els.dashboardStatus) return;
    if (!config.apiBase) {
      setStatus(els.dashboardStatus, 'Set the API base URL to load dashboards.', 'error');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Set the API base URL to load the map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Map unavailable';
      if (els.mapRemote) els.mapRemote.textContent = 'Remote: --';
      updateInsights({});
      renderFunnel({});
      renderTimeInStage({});
      clearDashboardDetail();
      state.dashboardEntries = [];
      state.mapDetails = null;
      state.calendarDetails = null;
      state.statusDetails = null;
      state.weekDetails = null;
      state.weekdayDetails = null;
      state.weeklySeries = [];
      buildWeekdayHeatmap(new Map());
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.dashboardStatus, 'Sign in to load your dashboards.', 'info');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Sign in to load the map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Sign in to view map';
      if (els.mapRemote) els.mapRemote.textContent = 'Remote: --';
      updateInsights({});
      renderFunnel({});
      renderTimeInStage({});
      clearDashboardDetail();
      state.dashboardEntries = [];
      state.mapDetails = null;
      state.calendarDetails = null;
      state.statusDetails = null;
      state.weekDetails = null;
      state.weekdayDetails = null;
      state.weeklySeries = [];
      buildWeekdayHeatmap(new Map());
      return;
    }
    const range = readRange();
    state.range = range;
    updateRangeInputs(range);
    const rangeLabel = formatRangeLabel(range.start, range.end);
    setStatus(els.dashboardStatus, 'Loading dashboards...', 'info');
    if (els.dashboard) els.dashboard.setAttribute('aria-busy', 'true');
    setOverlay(els.lineOverlay, 'Loading chart...');
    setOverlay(els.statusOverlay, 'Loading chart...');
    clearDashboardDetail();
    try {
      const query = buildQuery(range);
      const [summary, timeline, statuses, calendar, applications, funnel, timeInStage] = await Promise.all([
        requestJson(`/api/analytics/summary?${query}`),
        requestJson(`/api/analytics/applications-over-time?${query}`),
        requestJson(`/api/analytics/status-breakdown?${query}`),
        requestJson(`/api/analytics/calendar?${query}`),
        requestJson(`/api/applications?${query}`),
        requestJson(`/api/analytics/funnel?${query}`),
        requestJson(`/api/analytics/time-in-stage?${query}`)
      ]);
      const series = timeline.series || [];
      const statusSeries = statuses.statuses || [];
      const appItems = (Array.isArray(applications.items) ? applications.items : [])
        .map(item => normalizeEntry(item, 'application'));
      state.dashboardEntries = appItems;
      const detailData = buildDashboardDetails(appItems);
      state.calendarDetails = detailData.calendarDetails;
      state.statusDetails = detailData.statusDetails;
      state.weekDetails = detailData.weekDetails;
      state.weekdayDetails = detailData.weekdayDetails;
      updateKpis(summary);
      updateLineChart(series, rangeLabel);
      updateStatusChart(statusSeries);
      buildCalendar(calendar.days || [], rangeLabel, range, detailData.calendarDetails);
      buildWeekdayHeatmap(detailData.weekdayDetails);
      updateInsights(buildInsights(appItems));
      renderFunnel(funnel || {});
      renderTimeInStage(timeInStage || {});
      await updateMap(appItems, detailData);
      setOverlay(els.lineOverlay, series.length ? '' : 'No activity yet.');
      setOverlay(els.statusOverlay, statusSeries.length ? '' : 'No statuses yet.');
      setStatus(els.dashboardStatus, `Loaded ${summary.totalApplications || 0} applications.`, 'success');
    } catch (err) {
      console.error('Dashboard load failed', err);
      setOverlay(els.lineOverlay, 'Unable to load chart.');
      setOverlay(els.statusOverlay, 'Unable to load chart.');
      if (els.mapPlaceholder) els.mapPlaceholder.textContent = 'Unable to load map.';
      if (els.mapTotal) els.mapTotal.textContent = 'Map unavailable';
      if (els.mapRemote) els.mapRemote.textContent = 'Remote: --';
      updateInsights({});
      renderFunnel({});
      renderTimeInStage({});
      clearDashboardDetail();
      state.dashboardEntries = [];
      state.mapDetails = null;
      state.calendarDetails = null;
      state.statusDetails = null;
      state.weekDetails = null;
      state.weekdayDetails = null;
      state.weeklySeries = [];
      buildWeekdayHeatmap(new Map());
      setStatus(els.dashboardStatus, err?.message || 'Unable to load dashboards.', 'error');
    } finally {
      if (els.dashboard) els.dashboard.setAttribute('aria-busy', 'false');
    }
  };

  const normalizeEntry = (item, entryType) => ({
    ...item,
    recordType: entryType,
    entryType,
    statusDate: deriveStatusDate(item),
    tags: Array.isArray(item?.tags) ? item.tags : [],
    customFields: item?.customFields && typeof item.customFields === 'object' ? item.customFields : {}
  });

  const getEntryDateValue = (entry) => {
    const entryType = entry?.entryType || getEntryType(entry);
    return entryType === 'prospect' ? entry.captureDate : entry.appliedDate;
  };

  const getEntryDate = (entry) => {
    const raw = getEntryDateValue(entry);
    return parseDateInput(raw);
  };

  const updateEntryStatusFilter = (items = []) => {
    if (!els.entryFilterStatus) return;
    const current = els.entryFilterStatus.value || 'all';
    const statuses = new Set();
    items.forEach((item) => {
      statuses.add(getEntryStatusLabel(item));
    });
    const sorted = Array.from(statuses).sort((a, b) => a.localeCompare(b));
    els.entryFilterStatus.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = 'all';
    allOpt.textContent = 'All statuses';
    els.entryFilterStatus.appendChild(allOpt);
    sorted.forEach((status) => {
      const opt = document.createElement('option');
      opt.value = status.toLowerCase();
      opt.textContent = status;
      els.entryFilterStatus.appendChild(opt);
    });
    if ([...els.entryFilterStatus.options].some(opt => opt.value === current)) {
      els.entryFilterStatus.value = current;
    }
  };

  const updateEntrySourceFilter = (items = []) => {
    if (!els.entryFilterSource) return;
    const current = els.entryFilterSource.value || 'all';
    const sources = new Map();
    items.forEach((item) => {
      const source = (item.source || '').toString().trim();
      if (!source) return;
      const key = source.toLowerCase();
      if (!sources.has(key)) sources.set(key, source);
    });
    const sorted = Array.from(sources.entries())
      .sort((a, b) => a[1].localeCompare(b[1], 'en', { sensitivity: 'base' }));
    els.entryFilterSource.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = 'all';
    allOpt.textContent = 'All sources';
    els.entryFilterSource.appendChild(allOpt);
    sorted.forEach(([key, label]) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = label;
      els.entryFilterSource.appendChild(opt);
    });
    if ([...els.entryFilterSource.options].some(opt => opt.value === current)) {
      els.entryFilterSource.value = current;
    }
  };

  const updateEntryBatchFilter = (items = []) => {
    if (!els.entryFilterBatch) return;
    const current = els.entryFilterBatch.value || 'all';
    const batches = new Map();
    items.forEach((item) => {
      const batch = (item.batch || '').toString().trim();
      if (!batch) return;
      const key = batch.toLowerCase();
      if (!batches.has(key)) batches.set(key, batch);
    });
    const sorted = Array.from(batches.entries())
      .sort((a, b) => a[1].localeCompare(b[1], 'en', { sensitivity: 'base' }));
    els.entryFilterBatch.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = 'all';
    allOpt.textContent = 'All batches';
    els.entryFilterBatch.appendChild(allOpt);
    sorted.forEach(([key, label]) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = label;
      els.entryFilterBatch.appendChild(opt);
    });
    if ([...els.entryFilterBatch.options].some(opt => opt.value === current)) {
      els.entryFilterBatch.value = current;
    }
  };

  const matchesQuery = (entry, terms = []) => {
    if (!terms.length) return true;
    const tags = Array.isArray(entry.tags) ? entry.tags.join(' ') : '';
    const customFields = entry.customFields && typeof entry.customFields === 'object'
      ? Object.entries(entry.customFields).map(([key, value]) => `${key} ${value}`).join(' ')
      : '';
    const haystack = [
      entry.company,
      entry.title,
      entry.location,
      entry.source,
      entry.batch,
      entry.notes,
      entry.status,
      entry.followUpNote,
      tags,
      customFields
    ].join(' ').toLowerCase();
    return terms.every(term => haystack.includes(term));
  };

  const filterEntries = (items = []) => {
    const query = (els.entryFilterQuery?.value || '').trim().toLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);
    const type = (els.entryFilterType?.value || 'all').trim();
    const statusGroup = (els.entryFilterGroup?.value || 'all').trim();
    const status = (els.entryFilterStatus?.value || 'all').trim();
    const source = (els.entryFilterSource?.value || 'all').trim();
    const batch = (els.entryFilterBatch?.value || 'all').trim();
    const locationType = (els.entryFilterLocation?.value || 'all').trim();
    const start = parseDateInput(els.entryFilterStart?.value || '');
    const end = parseDateInput(els.entryFilterEnd?.value || '');
    const tagTerms = parseTagList(els.entryFilterTags?.value || '').map(tag => tag.toLowerCase());

    return items.filter((entry) => {
      const entryType = entry.entryType || getEntryType(entry);
      if (type !== 'all' && entryType !== type) return false;
      if (statusGroup !== 'all' && getEntryStatusGroup(entry) !== statusGroup) return false;
      const entryStatus = getEntryStatusLabel(entry);
      if (status !== 'all' && entryStatus.toLowerCase() !== status) return false;
      if (source !== 'all') {
        const entrySource = (entry.source || '').toString().trim().toLowerCase();
        if (entrySource !== source) return false;
      }
      if (batch !== 'all') {
        const entryBatch = (entry.batch || '').toString().trim().toLowerCase();
        if (entryBatch !== batch) return false;
      }
      if (tagTerms.length) {
        const entryTags = Array.isArray(entry.tags)
          ? entry.tags.map(tag => tag.toString().trim().toLowerCase())
          : [];
        if (!tagTerms.every(tag => entryTags.includes(tag))) return false;
      }
	      if (locationType !== 'all') {
	        const info = parseLocation(entry.location || '');
	        const hasLocation = Boolean(info.raw);
	        const isRemote = info.type === 'remote';
	        const isOnsite = info.type === 'onsite' || info.type === 'hybrid';
	        if (locationType === 'remote' && (!hasLocation || !isRemote)) return false;
	        if (locationType === 'onsite' && (!hasLocation || !isOnsite)) return false;
	      }
      if (!matchesQuery(entry, terms)) return false;
      if (start || end) {
        const dateValue = getEntryDate(entry);
        if (!dateValue) return false;
        if (start && dateValue < start) return false;
        if (end && dateValue > end) return false;
      }
      return true;
    });
  };

  const sortEntries = (items = []) => {
    const { key, direction } = state.entrySort;
    const multiplier = direction === 'asc' ? 1 : -1;
    const sorted = [...items];
    sorted.sort((a, b) => {
      let aVal = '';
      let bVal = '';
      if (key === 'date') {
        const aDate = getEntryDate(a);
        const bDate = getEntryDate(b);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return (aTime - bTime) * multiplier;
      }
      if (key === 'postingDate') {
        const aDate = parseDateInput(a.postingDate);
        const bDate = parseDateInput(b.postingDate);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return (aTime - bTime) * multiplier;
      }
      if (key === 'type') {
        aVal = a.entryType || getEntryType(a);
        bVal = b.entryType || getEntryType(b);
      } else if (key === 'company') {
        aVal = a.company || '';
        bVal = b.company || '';
      } else if (key === 'title') {
        aVal = a.title || '';
        bVal = b.title || '';
      } else if (key === 'status') {
        aVal = a.status || '';
        bVal = b.status || '';
	      } else if (key === 'location') {
	        aVal = formatLocationDisplay(a.location || '');
	        bVal = formatLocationDisplay(b.location || '');
	      } else if (key === 'source') {
	        aVal = a.source || '';
	        bVal = b.source || '';
      } else if (key === 'batch') {
        aVal = a.batch || '';
        bVal = b.batch || '';
      }
      return aVal.toString().localeCompare(bVal.toString(), 'en', { sensitivity: 'base' }) * multiplier;
    });
    return sorted;
  };

  const createEntryRow = (entry, { isExample = false } = {}) => {
    const row = document.createElement('div');
    row.className = 'jobtrack-table-row';
    if (isExample) row.classList.add('jobtrack-table-example');
    row.setAttribute('role', 'row');
    const entryType = entry.entryType || getEntryType(entry);
    const entryLabel = [entry.title, entry.company].filter(Boolean).join(' · ') || 'entry';
    if (entry.applicationId) {
      row.dataset.jobtrackRow = entry.applicationId;
      row.tabIndex = 0;
      row.setAttribute('aria-label', `View ${entryLabel}`);
    }

    const selectCell = document.createElement('div');
    selectCell.className = 'jobtrack-table-cell jobtrack-table-select';
    selectCell.dataset.label = 'Select';
    selectCell.dataset.cell = 'select';
    if (entry.applicationId) {
      const checkboxLabel = document.createElement('label');
      checkboxLabel.className = 'jobtrack-checkbox';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'jobtrack-checkbox-input';
      checkbox.dataset.jobtrackEntry = 'select';
      checkbox.dataset.id = entry.applicationId;
      checkbox.checked = state.selectedEntryIds.has(entry.applicationId);
      checkbox.setAttribute('aria-label', `Select ${entryLabel}`);
      checkboxLabel.appendChild(checkbox);
      selectCell.appendChild(checkboxLabel);
    }
    row.appendChild(selectCell);

    const typeCell = document.createElement('div');
    typeCell.className = 'jobtrack-table-cell';
    typeCell.dataset.label = 'Type';
    const pill = document.createElement('span');
    pill.className = 'jobtrack-pill';
    if (entryType === 'prospect') pill.dataset.tone = 'prospect';
    pill.textContent = entryType === 'prospect' ? 'Prospect' : 'Application';
    typeCell.appendChild(pill);
    row.appendChild(typeCell);

    const companyCell = document.createElement('div');
    companyCell.className = 'jobtrack-table-cell';
    companyCell.dataset.label = 'Company';
    companyCell.textContent = entry.company || '—';
    row.appendChild(companyCell);

    const titleCell = document.createElement('div');
    titleCell.className = 'jobtrack-table-cell';
    titleCell.dataset.label = 'Role';
    titleCell.textContent = entry.title || '—';
    row.appendChild(titleCell);

    const statusCell = document.createElement('div');
    statusCell.className = 'jobtrack-table-cell';
    statusCell.dataset.label = 'Status';
    const statusLabel = getEntryStatusLabel(entry);
    const statusPill = document.createElement('span');
    statusPill.className = 'jobtrack-status-pill';
    statusPill.dataset.tone = getStatusTone(statusLabel);
    statusPill.textContent = statusLabel;
    statusCell.appendChild(statusPill);
    row.appendChild(statusCell);

    const nextActionCell = document.createElement('div');
    nextActionCell.className = 'jobtrack-table-cell';
    nextActionCell.dataset.label = 'Next action';
    const action = getNextAction(entry);
    const actionPill = document.createElement('span');
    actionPill.className = 'jobtrack-action-pill';
    if (action.tone) actionPill.dataset.tone = action.tone;
    actionPill.textContent = action.label;
    nextActionCell.appendChild(actionPill);
    row.appendChild(nextActionCell);

	    const locationCell = document.createElement('div');
	    locationCell.className = 'jobtrack-table-cell';
	    locationCell.dataset.label = 'Location';
	    locationCell.textContent = formatLocationDisplay(entry.location || '') || '—';
	    row.appendChild(locationCell);

    const postingCell = document.createElement('div');
    postingCell.className = 'jobtrack-table-cell';
    postingCell.dataset.label = 'Posted';
    postingCell.textContent = entry.postingDate ? formatDateLabel(entry.postingDate) : '—';
    row.appendChild(postingCell);

    const dateCell = document.createElement('div');
    dateCell.className = 'jobtrack-table-cell';
    dateCell.dataset.label = entryType === 'prospect' ? 'Found date' : 'Applied date';
    const dateLabel = getEntryDateValue(entry);
    dateCell.textContent = dateLabel
      ? `${entryType === 'prospect' ? 'Found' : 'Applied'} ${formatDateLabel(dateLabel)}`
      : '—';
    row.appendChild(dateCell);

    const sourceCell = document.createElement('div');
    sourceCell.className = 'jobtrack-table-cell';
    sourceCell.dataset.label = 'Source';
    sourceCell.textContent = entry.source || '—';
    row.appendChild(sourceCell);

    const batchCell = document.createElement('div');
    batchCell.className = 'jobtrack-table-cell';
    batchCell.dataset.label = 'Batch';
    batchCell.textContent = entry.batch || '—';
    row.appendChild(batchCell);

    const tagsCell = document.createElement('div');
    tagsCell.className = 'jobtrack-table-cell';
    tagsCell.dataset.label = 'Tags';
    const tagValues = Array.isArray(entry.tags) ? entry.tags.filter(Boolean) : [];
    if (tagValues.length) {
      const wrap = document.createElement('div');
      wrap.className = 'jobtrack-tag-list';
      tagValues.slice(0, 3).forEach((tag) => {
        const pill = document.createElement('span');
        pill.className = 'jobtrack-tag';
        pill.textContent = tag;
        wrap.appendChild(pill);
      });
      if (tagValues.length > 3) {
        const more = document.createElement('span');
        more.className = 'jobtrack-tag';
        more.textContent = `+${tagValues.length - 3}`;
        wrap.appendChild(more);
      }
      tagsCell.appendChild(wrap);
    } else {
      tagsCell.textContent = '—';
    }
    row.appendChild(tagsCell);

    const actionsCell = document.createElement('div');
    actionsCell.className = 'jobtrack-table-cell jobtrack-table-actions';
    actionsCell.dataset.label = 'Actions';
    actionsCell.dataset.cell = 'actions';
    if (entry.applicationId) {
      const primaryRow = document.createElement('div');
      primaryRow.className = 'jobtrack-table-actions-row';

      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'btn-ghost';
      editBtn.dataset.jobtrackEntry = 'edit';
      editBtn.dataset.id = entry.applicationId;
      editBtn.textContent = 'Edit';
      primaryRow.appendChild(editBtn);

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'btn-ghost';
      deleteBtn.dataset.jobtrackEntry = 'delete';
      deleteBtn.dataset.id = entry.applicationId;
      deleteBtn.textContent = 'Delete';
      primaryRow.appendChild(deleteBtn);

      if (entryType === 'prospect') {
        const applyBtn = document.createElement('button');
        applyBtn.type = 'button';
        applyBtn.className = 'btn-ghost';
        applyBtn.dataset.jobtrackEntry = 'apply';
        applyBtn.dataset.id = entry.applicationId;
        applyBtn.textContent = 'Apply';
        primaryRow.appendChild(applyBtn);
      }
      actionsCell.appendChild(primaryRow);

      const attachments = Array.isArray(entry.attachments) ? entry.attachments : [];
      const zipRow = document.createElement('div');
      zipRow.className = 'jobtrack-table-actions-row';
      const zipBtn = document.createElement('button');
      zipBtn.type = 'button';
      zipBtn.className = 'btn-ghost jobtrack-attachment-btn';
      zipBtn.dataset.jobtrackEntry = 'download-zip';
      zipBtn.dataset.id = entry.applicationId;
      zipBtn.textContent = 'Download ZIP';
      if (!attachments.length) {
        zipBtn.disabled = true;
        zipBtn.setAttribute('aria-disabled', 'true');
        zipBtn.title = 'No attachments to zip';
      } else {
        zipBtn.title = 'Download all attachments as ZIP';
      }
      zipRow.appendChild(zipBtn);
      actionsCell.appendChild(zipRow);
    } else {
      actionsCell.textContent = '—';
    }
    row.appendChild(actionsCell);
    return row;
  };

  const renderEntryList = (items = [], emptyLabel = 'No entries yet.') => {
    if (!els.entryList) return;
    els.entryList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'jobtrack-table-empty';
      empty.textContent = `${emptyLabel} Example row below shows what goes where.`;
      els.entryList.appendChild(empty);
      const example = createEntryRow({
        entryType: 'application',
        company: 'Acme Corp',
        title: 'Data Analyst',
        status: 'Interview',
        statusDate: '2025-03-10',
        location: 'Remote - US',
        postingDate: '2025-03-01',
        appliedDate: '2025-03-05',
        source: 'LinkedIn',
        batch: 'March outreach'
      }, { isExample: true });
      els.entryList.appendChild(example);
      return;
    }
    items.forEach((entry) => {
      els.entryList.appendChild(createEntryRow(entry));
    });
  };

  const buildProspectApplyAttachments = (form) => {
    if (!form) return [];
    const resume = form.querySelector('[data-jobtrack-prospect="resume"]')?.files?.[0];
    const cover = form.querySelector('[data-jobtrack-prospect="cover"]')?.files?.[0];
    const attachments = [];
    if (resume) attachments.push({ file: resume, kind: 'resume' });
    if (cover) attachments.push({ file: cover, kind: 'cover-letter' });
    return attachments;
  };

  const renderProspectReviewList = (items = []) => {
    if (!els.prospectReviewList) return;
    els.prospectReviewList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('p');
      empty.className = 'jobtrack-prospect-empty';
      empty.textContent = 'No active prospects in the queue.';
      els.prospectReviewList.appendChild(empty);
      return;
    }
    const list = document.createElement('ul');
    list.className = 'jobtrack-prospect-list';
    items.forEach((entry, index) => {
      const promptContext = buildPromptContextFromEntry(entry);
      const item = document.createElement('li');
      item.className = 'jobtrack-prospect-item';

      const header = document.createElement('div');
      header.className = 'jobtrack-prospect-head';

      const indexBadge = document.createElement('span');
      indexBadge.className = 'jobtrack-prospect-index';
      indexBadge.textContent = `${index + 1}`;
      header.appendChild(indexBadge);

      const title = document.createElement('div');
      title.className = 'jobtrack-prospect-title';
      title.textContent = [entry.title, entry.company].filter(Boolean).join(' · ') || 'Prospect';
      header.appendChild(title);

      const posted = document.createElement('span');
      posted.className = 'jobtrack-prospect-posted';
      const postedLabel = entry.postingDate && parseDateInput(entry.postingDate)
        ? formatDateLabel(entry.postingDate)
        : 'Date unknown';
      posted.textContent = `Posted ${postedLabel}`;
      header.appendChild(posted);

      item.appendChild(header);

      const meta = document.createElement('div');
      meta.className = 'jobtrack-prospect-meta';
      meta.textContent = formatProspectMeta(entry) || 'Prospect details';
      item.appendChild(meta);

      if (entry.jobUrl) {
        const link = document.createElement('a');
        link.className = 'jobtrack-prospect-link';
        link.href = normalizeUrl(entry.jobUrl);
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = 'Open job posting';
        item.appendChild(link);
      }

      if (entry.notes) {
        const notes = document.createElement('p');
        notes.className = 'jobtrack-prospect-notes';
        notes.textContent = entry.notes;
        item.appendChild(notes);
      }

      const actions = document.createElement('div');
      actions.className = 'jobtrack-prospect-actions';

      const detailBtn = document.createElement('button');
      detailBtn.type = 'button';
      detailBtn.className = 'btn-ghost jobtrack-prospect-action';
      detailBtn.textContent = 'View details';
      detailBtn.addEventListener('click', () => openDetailModal(entry));
      actions.appendChild(detailBtn);

      const rejectBtn = document.createElement('button');
      rejectBtn.type = 'button';
      rejectBtn.className = 'btn-ghost jobtrack-prospect-action';
      rejectBtn.textContent = 'Reject + archive';
      rejectBtn.addEventListener('click', () => archiveProspectEntry(entry));
      actions.appendChild(rejectBtn);

      item.appendChild(actions);

      const applyDetails = document.createElement('details');
      applyDetails.className = 'jobtrack-prospect-apply';
      const applySummary = document.createElement('summary');
      applySummary.textContent = 'Apply to this role';
      applyDetails.appendChild(applySummary);

      const applyForm = document.createElement('form');
      applyForm.className = 'jobtrack-prospect-apply-form';
      applyForm.noValidate = true;

      const fields = document.createElement('div');
      fields.className = 'jobtrack-prospect-apply-fields';

      const dateField = document.createElement('div');
      dateField.className = 'jobtrack-field';
      const dateLabel = document.createElement('label');
      dateLabel.className = 'jobtrack-label';
      const dateId = `jobtrack-prospect-apply-date-${(entry.applicationId || 'prospect').toString().replace(/[^a-z0-9_-]/gi, '')}`;
      dateLabel.setAttribute('for', dateId);
      dateLabel.textContent = 'Applied date';
      const dateInput = document.createElement('input');
      dateInput.type = 'date';
      dateInput.className = 'jobtrack-input';
      dateInput.id = dateId;
      dateInput.required = true;
      dateInput.value = formatDateInput(new Date());
      dateField.appendChild(dateLabel);
      dateField.appendChild(dateInput);
      fields.appendChild(dateField);

      const resumeField = document.createElement('div');
      resumeField.className = 'jobtrack-field';
      const resumeLabel = document.createElement('label');
      resumeLabel.className = 'jobtrack-label';
      const resumeId = `jobtrack-prospect-apply-resume-${(entry.applicationId || 'prospect').toString().replace(/[^a-z0-9_-]/gi, '')}`;
      resumeLabel.setAttribute('for', resumeId);
      resumeLabel.textContent = 'Resume (optional)';
      const resumeInput = document.createElement('input');
      resumeInput.type = 'file';
      resumeInput.className = 'jobtrack-input jobtrack-file-input';
      resumeInput.id = resumeId;
      resumeInput.dataset.jobtrackProspect = 'resume';
      resumeInput.accept = '.pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      resumeField.appendChild(resumeLabel);
      resumeField.appendChild(resumeInput);
      const resumePromptActions = document.createElement('div');
      resumePromptActions.className = 'jobtrack-prompt-actions';
      const resumePromptBtn = document.createElement('button');
      resumePromptBtn.type = 'button';
      resumePromptBtn.className = 'btn-ghost jobtrack-prompt-action';
      resumePromptBtn.textContent = 'Download resume prompt';
      resumePromptBtn.addEventListener('click', () => downloadEntryPrompt(promptContext, 'resume'));
      resumePromptActions.appendChild(resumePromptBtn);
      resumeField.appendChild(resumePromptActions);
      fields.appendChild(resumeField);

      const coverField = document.createElement('div');
      coverField.className = 'jobtrack-field';
      const coverLabel = document.createElement('label');
      coverLabel.className = 'jobtrack-label';
      const coverId = `jobtrack-prospect-apply-cover-${(entry.applicationId || 'prospect').toString().replace(/[^a-z0-9_-]/gi, '')}`;
      coverLabel.setAttribute('for', coverId);
      coverLabel.textContent = 'Cover letter (optional)';
      const coverInput = document.createElement('input');
      coverInput.type = 'file';
      coverInput.className = 'jobtrack-input jobtrack-file-input';
      coverInput.id = coverId;
      coverInput.dataset.jobtrackProspect = 'cover';
      coverInput.accept = '.pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      coverField.appendChild(coverLabel);
      coverField.appendChild(coverInput);
      const coverPromptActions = document.createElement('div');
      coverPromptActions.className = 'jobtrack-prompt-actions';
      const coverPromptBtn = document.createElement('button');
      coverPromptBtn.type = 'button';
      coverPromptBtn.className = 'btn-ghost jobtrack-prompt-action';
      coverPromptBtn.textContent = 'Download cover letter prompt';
      coverPromptBtn.addEventListener('click', () => downloadEntryPrompt(promptContext, 'cover'));
      coverPromptActions.appendChild(coverPromptBtn);
      coverField.appendChild(coverPromptActions);
      fields.appendChild(coverField);

      applyForm.appendChild(fields);

      const applyHelp = document.createElement('p');
      applyHelp.className = 'jobtrack-help';
      applyHelp.textContent = 'Optional files are stored with the new application.';
      applyForm.appendChild(applyHelp);

      const applyActions = document.createElement('div');
      applyActions.className = 'jobtrack-prospect-apply-actions';
      const applyBtn = document.createElement('button');
      applyBtn.type = 'submit';
      applyBtn.className = 'btn-primary';
      applyBtn.textContent = 'Convert and apply';
      applyActions.appendChild(applyBtn);
      applyForm.appendChild(applyActions);

      const status = document.createElement('p');
      status.className = 'jobtrack-form-status';
      status.dataset.jobtrackProspectStatus = entry.applicationId || '';
      status.textContent = 'Ready to apply.';
      applyForm.appendChild(status);

      applyForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const appliedDate = (dateInput.value || '').toString().trim();
        if (!appliedDate) {
          setStatus(status, 'Applied date is required.', 'error');
          return;
        }
        if (!parseDateInput(appliedDate)) {
          setStatus(status, 'Applied date must be valid (YYYY-MM-DD).', 'error');
          return;
        }
        const attachments = buildProspectApplyAttachments(applyForm);
        applyBtn.disabled = true;
        applyBtn.setAttribute('aria-disabled', 'true');
        const ok = await convertProspectToApplicationWithAttachments(entry, appliedDate, attachments, status);
        if (!ok) {
          applyBtn.disabled = false;
          applyBtn.setAttribute('aria-disabled', 'false');
        }
      });

      applyDetails.appendChild(applyForm);
      item.appendChild(applyDetails);
      list.appendChild(item);
    });
    els.prospectReviewList.appendChild(list);
  };

  const updateSortIndicators = () => {
    if (els.entrySortButtons.length) {
      els.entrySortButtons.forEach((button) => {
        const key = button.dataset.jobtrackSort;
        if (key === state.entrySort.key) {
          button.setAttribute('aria-sort', state.entrySort.direction === 'asc' ? 'ascending' : 'descending');
        } else {
          button.setAttribute('aria-sort', 'none');
        }
      });
    }
    if (els.entrySortSelect) {
      const exact = `${state.entrySort.key}-${state.entrySort.direction}`;
      const fallback = `${state.entrySort.key}-asc`;
      const options = [...els.entrySortSelect.options].map(opt => opt.value);
      if (options.includes(exact)) {
        els.entrySortSelect.value = exact;
      } else if (options.includes(fallback)) {
        els.entrySortSelect.value = fallback;
      }
    }
  };

  const setVisibleEntryIds = (items = []) => {
    state.visibleEntryIds = items
      .map(item => item.applicationId)
      .filter(Boolean);
  };

  const updateSelectAllState = () => {
    if (!els.entrySelectAll) return;
    const visible = state.visibleEntryIds || [];
    if (!visible.length) {
      els.entrySelectAll.checked = false;
      els.entrySelectAll.indeterminate = false;
      els.entrySelectAll.disabled = true;
      els.entrySelectAll.setAttribute('aria-disabled', 'true');
      return;
    }
    els.entrySelectAll.disabled = false;
    els.entrySelectAll.setAttribute('aria-disabled', 'false');
    const selectedVisible = visible.filter(id => state.selectedEntryIds.has(id)).length;
    els.entrySelectAll.checked = selectedVisible === visible.length;
    els.entrySelectAll.indeterminate = selectedVisible > 0 && selectedVisible < visible.length;
  };

  const updateEntrySelectionUI = () => {
    const selectedIds = Array.from(state.selectedEntryIds);
    const selectedCount = selectedIds.length;
    const selectedApplications = selectedIds.filter(id => {
      const item = state.entryItems.get(id);
      return item?.entryType === 'application';
    }).length;
    if (els.entrySelectedCount) {
      els.entrySelectedCount.textContent = `${selectedCount} selected`;
    }
    if (els.entryBulkDelete) {
      els.entryBulkDelete.disabled = selectedCount === 0;
      els.entryBulkDelete.setAttribute('aria-disabled', selectedCount === 0 ? 'true' : 'false');
    }
    if (els.bulkStatusApply) {
      const disabled = selectedApplications === 0;
      els.bulkStatusApply.disabled = disabled;
      els.bulkStatusApply.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }
    if (els.bulkStatusSelect) {
      const disabled = selectedApplications === 0;
      els.bulkStatusSelect.disabled = disabled;
      els.bulkStatusSelect.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }
    if (els.bulkStatusDate) {
      const disabled = selectedApplications === 0;
      els.bulkStatusDate.disabled = disabled;
      els.bulkStatusDate.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }
    updateSelectAllState();
  };

  const updateEntryCount = (visibleCount, totalCount) => {
    if (!els.entryCount) return;
    if (!totalCount) {
      els.entryCount.textContent = '0 entries';
      return;
    }
    const noun = totalCount === 1 ? 'entry' : 'entries';
    const label = visibleCount === totalCount
      ? `${totalCount} ${noun}`
      : `Showing ${visibleCount} of ${totalCount} ${noun}`;
    els.entryCount.textContent = label;
  };

  const setEntryView = (view) => {
    const nextView = view === 'cards' ? 'cards' : 'table';
    state.entryView = nextView;
    if (els.entryViewWrap) els.entryViewWrap.dataset.view = nextView;
    if (els.entryViewInputs.length) {
      els.entryViewInputs.forEach((input) => {
        input.checked = input.value === nextView;
      });
    }
    try {
      localStorage.setItem(ENTRY_VIEW_KEY, nextView);
    } catch {}
  };

  const initEntryView = () => {
    if (!els.entryViewWrap) return;
    let stored = '';
    try {
      stored = localStorage.getItem(ENTRY_VIEW_KEY) || '';
    } catch {}
    setEntryView(stored || state.entryView);
    if (els.entryViewInputs.length) {
      els.entryViewInputs.forEach((input) => {
        input.addEventListener('change', () => setEntryView(input.value));
      });
    }
  };

  const applyEntryFilters = () => {
    const filtered = filterEntries(state.entries);
    const sorted = sortEntries(filtered);
    const emptyLabel = state.entries.length ? 'No entries match your filters yet.' : 'No entries yet.';
    renderEntryList(sorted, emptyLabel);
    setVisibleEntryIds(sorted);
    updateSortIndicators();
    updateEntrySelectionUI();
    updateEntryCount(sorted.length, state.entries.length);
  };

  const buildSavedViewFilters = () => ({
    query: (els.entryFilterQuery?.value || '').toString().trim(),
    statusGroup: (els.entryFilterGroup?.value || 'all').toString().trim(),
    type: (els.entryFilterType?.value || 'all').toString().trim(),
    status: (els.entryFilterStatus?.value || 'all').toString().trim(),
    source: (els.entryFilterSource?.value || 'all').toString().trim(),
    batch: (els.entryFilterBatch?.value || 'all').toString().trim(),
    location: (els.entryFilterLocation?.value || 'all').toString().trim(),
    start: (els.entryFilterStart?.value || '').toString().trim(),
    end: (els.entryFilterEnd?.value || '').toString().trim(),
    tags: parseTagInput(els.entryFilterTags?.value || ''),
    sort: els.entrySortSelect?.value || `${state.entrySort.key}-${state.entrySort.direction}`,
    view: state.entryView
  });

  const updateSavedViewActions = () => {
    if (!els.savedViewDelete) return;
    const hasSelection = Boolean(state.activeViewId);
    els.savedViewDelete.disabled = !hasSelection;
    els.savedViewDelete.setAttribute('aria-disabled', hasSelection ? 'false' : 'true');
  };

  const renderSavedViews = () => {
    if (!els.savedViewSelect) return;
    els.savedViewSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose a saved view';
    els.savedViewSelect.appendChild(placeholder);
    state.savedViews.forEach((view) => {
      const option = document.createElement('option');
      option.value = view.applicationId || '';
      option.textContent = view.name || 'Saved view';
      els.savedViewSelect.appendChild(option);
    });
    if (state.activeViewId) {
      els.savedViewSelect.value = state.activeViewId;
    }
    updateSavedViewActions();
  };

  const applySavedView = (view) => {
    if (!view) return;
    const filters = view.filters || {};
    if (els.entryFilterQuery) els.entryFilterQuery.value = filters.query || '';
    if (els.entryFilterGroup) els.entryFilterGroup.value = filters.statusGroup || 'all';
    if (els.entryFilterType) els.entryFilterType.value = filters.type || 'all';
    if (els.entryFilterStatus) els.entryFilterStatus.value = filters.status || 'all';
    if (els.entryFilterSource) els.entryFilterSource.value = filters.source || 'all';
    if (els.entryFilterBatch) els.entryFilterBatch.value = filters.batch || 'all';
    if (els.entryFilterLocation) els.entryFilterLocation.value = filters.location || 'all';
    if (els.entryFilterStart) els.entryFilterStart.value = filters.start || '';
    if (els.entryFilterEnd) els.entryFilterEnd.value = filters.end || '';
    if (els.entryFilterTags) {
      const tags = Array.isArray(filters.tags) ? filters.tags : parseTagInput(filters.tags || '');
      els.entryFilterTags.value = formatTagInput(tags);
    }
    if (filters.sort) {
      const [key, direction] = filters.sort.split('-');
      state.entrySort.key = key || 'date';
      state.entrySort.direction = direction === 'asc' ? 'asc' : 'desc';
    }
    if (filters.view) {
      setEntryView(filters.view);
    }
    applyEntryFilters();
  };

  const loadSavedViews = async () => {
    if (!els.savedViewSelect) return;
    if (!config.apiBase || !authIsValid(state.auth)) {
      state.savedViews = [];
      state.activeViewId = '';
      renderSavedViews();
      if (els.savedViewStatus) {
        setStatus(els.savedViewStatus, 'Sign in to use saved views.', 'info');
      }
      return;
    }
    try {
      const data = await requestJson('/api/views');
      state.savedViews = Array.isArray(data.items) ? data.items : [];
      if (state.activeViewId && !state.savedViews.find(view => view.applicationId === state.activeViewId)) {
        state.activeViewId = '';
      }
      renderSavedViews();
      if (els.savedViewStatus) setStatus(els.savedViewStatus, '', '');
    } catch (err) {
      state.savedViews = [];
      state.activeViewId = '';
      renderSavedViews();
      if (els.savedViewStatus) {
        setStatus(els.savedViewStatus, err?.message || 'Unable to load saved views.', 'error');
      }
    }
  };

  const saveCurrentView = async () => {
    if (!els.savedViewName) return;
    const name = (els.savedViewName.value || '').toString().trim();
    if (!name) {
      setStatus(els.savedViewStatus, 'Name the view before saving.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.savedViewStatus, 'Sign in to save views.', 'error');
      return;
    }
    if (!config.apiBase) {
      setStatus(els.savedViewStatus, 'Set the API base URL to save views.', 'error');
      return;
    }
    try {
      setStatus(els.savedViewStatus, 'Saving view...', 'info');
      const created = await requestJson('/api/views', {
        method: 'POST',
        body: { name, filters: buildSavedViewFilters() }
      });
      if (created?.applicationId) {
        state.savedViews = [...state.savedViews, created];
        state.activeViewId = created.applicationId;
        renderSavedViews();
        els.savedViewName.value = '';
        setStatus(els.savedViewStatus, 'View saved.', 'success');
      } else {
        setStatus(els.savedViewStatus, 'View saved.', 'success');
        await loadSavedViews();
      }
    } catch (err) {
      setStatus(els.savedViewStatus, err?.message || 'Unable to save view.', 'error');
    }
  };

  const deleteSavedView = async () => {
    if (!state.activeViewId) return;
    if (!confirmAction('Delete this saved view?')) return;
    if (!authIsValid(state.auth)) {
      setStatus(els.savedViewStatus, 'Sign in to delete views.', 'error');
      return;
    }
    if (!config.apiBase) {
      setStatus(els.savedViewStatus, 'Set the API base URL to delete views.', 'error');
      return;
    }
    try {
      setStatus(els.savedViewStatus, 'Deleting view...', 'info');
      await requestJson(`/api/views/${encodeURIComponent(state.activeViewId)}`, { method: 'DELETE' });
      state.savedViews = state.savedViews.filter(view => view.applicationId !== state.activeViewId);
      state.activeViewId = '';
      renderSavedViews();
      setStatus(els.savedViewStatus, 'View deleted.', 'success');
    } catch (err) {
      setStatus(els.savedViewStatus, err?.message || 'Unable to delete view.', 'error');
    }
  };

  const renderFollowups = (items = []) => {
    if (!els.followupList) return;
    els.followupList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('p');
      empty.className = 'jobtrack-form-status';
      empty.textContent = 'No follow-ups due in this window.';
      els.followupList.appendChild(empty);
      return;
    }
    items.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'jobtrack-followup-item';
      if (item.actionTone) row.dataset.tone = item.actionTone;

      const info = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'jobtrack-followup-title';
      title.textContent = [item.title, item.company].filter(Boolean).join(' · ') || 'Follow-up';
      const meta = document.createElement('div');
      meta.className = 'jobtrack-followup-meta';
      const dueLabel = item.dueDate ? formatDateLabel(item.dueDate) : 'Soon';
      const duePrefix = item.overdue ? 'Overdue' : 'Due';
      const statusLabel = item.status ? ` · ${item.status}` : '';
      const noteLabel = item.followUpNote ? ` · ${item.followUpNote}` : '';
      meta.textContent = `${item.actionLabel || 'Follow up'} · ${duePrefix} ${dueLabel}${statusLabel}${noteLabel}`;
      info.appendChild(title);
      info.appendChild(meta);

      const actions = document.createElement('div');
      actions.className = 'jobtrack-followup-actions';
      const viewBtn = document.createElement('button');
      viewBtn.type = 'button';
      viewBtn.className = 'btn-ghost';
      viewBtn.textContent = 'View';
      viewBtn.addEventListener('click', () => {
        const entry = state.entryItems.get(item.applicationId);
        if (entry) {
          openDetailModal(entry);
          return;
        }
        setStatus(els.followupStatus, 'Entry details are not loaded yet.', 'info');
      });
      actions.appendChild(viewBtn);
      if (item.jobUrl) {
        const linkBtn = document.createElement('a');
        linkBtn.className = 'btn-ghost';
        linkBtn.href = normalizeUrl(item.jobUrl);
        linkBtn.target = '_blank';
        linkBtn.rel = 'noopener noreferrer';
        linkBtn.textContent = 'Job link';
        actions.appendChild(linkBtn);
      }

      row.appendChild(info);
      row.appendChild(actions);
      els.followupList.appendChild(row);
    });
  };

  const refreshFollowups = async () => {
    if (!els.followupStatus) return;
    if (!config.apiBase) {
      renderFollowups([]);
      setStatus(els.followupStatus, 'Set the API base URL to load follow-ups.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      renderFollowups([]);
      setStatus(els.followupStatus, 'Sign in to load follow-ups.', 'info');
      return;
    }
    const start = formatDateInput(new Date());
    const end = formatDateInput(addDays(new Date(), FOLLOWUP_RANGE_DAYS));
    try {
      setStatus(els.followupStatus, 'Loading follow-ups...', 'info');
      const data = await requestJson(`/api/analytics/followups?start=${start}&end=${end}&includeOverdue=true`);
      const items = Array.isArray(data.items) ? data.items : [];
      state.followups = items;
      renderFollowups(items);
      setStatus(els.followupStatus, items.length ? `Loaded ${items.length} follow-up${items.length === 1 ? '' : 's'}.` : 'No follow-ups due.', 'success');
    } catch (err) {
      renderFollowups([]);
      setStatus(els.followupStatus, err?.message || 'Unable to load follow-ups.', 'error');
    }
  };

	  const refreshEntries = async () => {
	    if (!els.entryList) return;
	    state.entriesLoaded = false;
	    if (!config.apiBase) {
	      storeEntries([]);
	      updateProspectDashboard();
	      state.selectedEntryIds.clear();
	      renderEntryList([], 'Set the API base URL to load entries.');
	      renderProspectReviewList([]);
	      if (els.prospectReviewStatus) {
	        setStatus(els.prospectReviewStatus, 'Set the API base URL to load prospects.', 'error');
      }
      updateEntrySelectionUI();
      updateEntryCount(0, 0);
      loadSavedViews();
      refreshFollowups();
      return;
	    }
	    if (!authIsValid(state.auth)) {
	      storeEntries([]);
	      updateProspectDashboard();
	      state.selectedEntryIds.clear();
	      renderEntryList([], 'Sign in to load your entries.');
	      renderProspectReviewList([]);
      if (els.prospectReviewStatus) {
        setStatus(els.prospectReviewStatus, 'Sign in to load prospects.', 'info');
      }
      updateEntrySelectionUI();
      updateEntryCount(0, 0);
      loadSavedViews();
      refreshFollowups();
      return;
    }
	    try {
	      setStatus(els.entryListStatus, 'Loading entries...', 'info');
	      const [apps, prospects] = await Promise.all([
	        requestJson('/api/applications'),
	        requestJson('/api/prospects')
	      ]);
      const appItems = (apps.items || []).map(item => normalizeEntry(item, 'application'));
      const prospectItems = (prospects.items || []).map(item => normalizeEntry(item, 'prospect'));
      const sortedProspects = [...prospectItems].sort((a, b) => {
        const aDate = getProspectSortDate(a);
        const bDate = getProspectSortDate(b);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        if (aTime !== bTime) return bTime - aTime;
        return getEntryLabel(a).localeCompare(getEntryLabel(b), 'en', { sensitivity: 'base' });
      });
	      const items = [...appItems, ...prospectItems];
	      storeEntries(items);
	      state.entriesLoaded = true;
	      updateProspectDashboard();
	      state.selectedEntryIds.clear();
	      updateEntryStatusFilter(items);
	      updateEntrySourceFilter(items);
	      updateEntryBatchFilter(items);
	      applyEntryFilters();
      const reviewProspects = sortedProspects.filter((entry) => getEntryStatusGroup(entry) === 'active');
      renderProspectReviewList(reviewProspects);
      if (els.prospectReviewStatus) {
        const label = reviewProspects.length === 1 ? '1 prospect' : `${reviewProspects.length} prospects`;
        setStatus(els.prospectReviewStatus, `Loaded ${label} in the queue.`, 'success');
      }
      setStatus(els.entryListStatus, `Loaded ${items.length} entries.`, 'success');
      loadSavedViews();
      refreshFollowups();
	    } catch (err) {
	      console.error('Entry load failed', err);
	      storeEntries([]);
	      state.entriesLoaded = true;
	      updateProspectDashboard();
	      renderEntryList([], 'Unable to load entries.');
	      renderProspectReviewList([]);
	      updateEntrySelectionUI();
	      updateEntryCount(0, 0);
      setStatus(els.entryListStatus, err?.message || 'Unable to load entries.', 'error');
      if (els.prospectReviewStatus) {
        setStatus(els.prospectReviewStatus, err?.message || 'Unable to load prospects.', 'error');
      }
      loadSavedViews();
      refreshFollowups();
    }
  };

  const initEntryList = () => {
    initEntryView();
    if (els.savedViewSelect) {
      els.savedViewSelect.addEventListener('change', () => {
        const viewId = els.savedViewSelect.value || '';
        state.activeViewId = viewId;
        updateSavedViewActions();
        if (!viewId) return;
        const view = state.savedViews.find(item => item.applicationId === viewId);
        if (view) applySavedView(view);
      });
    }
    if (els.savedViewSave) {
      els.savedViewSave.addEventListener('click', () => saveCurrentView());
    }
    if (els.savedViewDelete) {
      els.savedViewDelete.addEventListener('click', () => deleteSavedView());
    }
    if (els.followupRefresh) {
      els.followupRefresh.addEventListener('click', () => refreshFollowups());
    }
    if (els.entriesRefresh) {
      els.entriesRefresh.addEventListener('click', () => refreshEntries());
    }
    if (els.bulkStatusDate && !els.bulkStatusDate.value) {
      els.bulkStatusDate.value = formatDateInput(new Date());
    }
    if (els.bulkStatusApply) {
      els.bulkStatusApply.addEventListener('click', () => applyBulkStatus());
    }
    if (els.entryFilterQuery) {
      els.entryFilterQuery.addEventListener('input', () => applyEntryFilters());
    }
    if (els.entryFilterGroup) {
      els.entryFilterGroup.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterType) {
      els.entryFilterType.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterStatus) {
      els.entryFilterStatus.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterSource) {
      els.entryFilterSource.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterBatch) {
      els.entryFilterBatch.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterLocation) {
      els.entryFilterLocation.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterTags) {
      els.entryFilterTags.addEventListener('input', () => applyEntryFilters());
    }
    if (els.entryFilterStart) {
      els.entryFilterStart.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entryFilterEnd) {
      els.entryFilterEnd.addEventListener('change', () => applyEntryFilters());
    }
    if (els.entrySortSelect) {
      els.entrySortSelect.addEventListener('change', () => {
        const [key, direction] = (els.entrySortSelect.value || 'date-desc').split('-');
        state.entrySort.key = key || 'date';
        state.entrySort.direction = direction === 'asc' ? 'asc' : 'desc';
        applyEntryFilters();
      });
    }
    if (els.entryFilterReset) {
      els.entryFilterReset.addEventListener('click', () => {
        if (els.entryFilterQuery) els.entryFilterQuery.value = '';
        if (els.entryFilterGroup) els.entryFilterGroup.value = 'all';
        if (els.entryFilterType) els.entryFilterType.value = 'all';
        if (els.entryFilterStatus) els.entryFilterStatus.value = 'all';
        if (els.entryFilterSource) els.entryFilterSource.value = 'all';
        if (els.entryFilterBatch) els.entryFilterBatch.value = 'all';
        if (els.entryFilterLocation) els.entryFilterLocation.value = 'all';
        if (els.entryFilterTags) els.entryFilterTags.value = '';
        if (els.entryFilterStart) els.entryFilterStart.value = '';
        if (els.entryFilterEnd) els.entryFilterEnd.value = '';
        applyEntryFilters();
      });
    }
    if (els.entrySortButtons.length) {
      els.entrySortButtons.forEach((button) => {
        button.addEventListener('click', () => {
          const key = button.dataset.jobtrackSort;
          if (!key) return;
          if (state.entrySort.key === key) {
            state.entrySort.direction = state.entrySort.direction === 'asc' ? 'desc' : 'asc';
          } else {
            state.entrySort.key = key;
            state.entrySort.direction = key === 'date' ? 'desc' : 'asc';
          }
          applyEntryFilters();
        });
      });
    }
    if (els.entryUndoAction) {
      els.entryUndoAction.addEventListener('click', () => restoreDeletedEntry());
    }
    if (els.entrySelectAll) {
      els.entrySelectAll.addEventListener('change', () => {
        const visibleIds = state.visibleEntryIds || [];
        if (!visibleIds.length) return;
        if (els.entrySelectAll.checked) {
          visibleIds.forEach(id => state.selectedEntryIds.add(id));
        } else {
          visibleIds.forEach(id => state.selectedEntryIds.delete(id));
        }
        updateEntrySelectionUI();
        applyEntryFilters();
      });
    }
    if (els.entryBulkDelete) {
      els.entryBulkDelete.addEventListener('click', () => {
        deleteEntriesBulk(Array.from(state.selectedEntryIds));
      });
    }
    if (els.entryList) {
      els.entryList.addEventListener('change', (event) => {
        const input = event.target.closest('input[data-jobtrack-entry="select"]');
        if (!input) return;
        const entryId = input.dataset.id;
        if (!entryId) return;
        if (input.checked) {
          state.selectedEntryIds.add(entryId);
        } else {
          state.selectedEntryIds.delete(entryId);
        }
        updateEntrySelectionUI();
      });
      els.entryList.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-jobtrack-entry]');
        if (button) {
          const action = button.dataset.jobtrackEntry;
          if (action === 'download-zip') {
            const entryId = button.dataset.id;
            if (!entryId) return;
            downloadEntryZip(entryId);
            return;
          }
          const entryId = button.dataset.id;
          if (!entryId) return;
          if (action === 'edit') {
            const item = state.entryItems.get(entryId);
            if (item) setEntryEditMode(item);
            return;
          }
          if (action === 'apply') {
            const item = state.entryItems.get(entryId);
            if (item) openDetailModal(item, '[data-jobtrack="apply-date-input"]');
            return;
          }
          if (action === 'delete') {
            deleteEntry(entryId);
          }
          return;
        }
        if (event.target.closest('input, label, a')) return;
        const row = event.target.closest('.jobtrack-table-row');
        if (!row || !els.entryList.contains(row)) return;
        const entryId = row.dataset.jobtrackRow;
        if (!entryId) return;
        const entry = state.entryItems.get(entryId);
        if (entry) openDetailModal(entry);
      });
      els.entryList.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        if (event.target.closest('button, input, label, a')) return;
        const row = event.target.closest('.jobtrack-table-row');
        if (!row || !els.entryList.contains(row)) return;
        const entryId = row.dataset.jobtrackRow;
        if (!entryId) return;
        event.preventDefault();
        const entry = state.entryItems.get(entryId);
        if (entry) openDetailModal(entry);
      });
    }
  };

  const initProspectReview = () => {
    if (els.prospectReviewRefresh) {
      els.prospectReviewRefresh.addEventListener('click', () => refreshEntries());
    }
  };

  const initExport = () => {
    if (!els.exportForm) return;
    const range = defaultRange();
    if (els.exportStart && !els.exportStart.value) els.exportStart.value = formatDateInput(range.start);
    if (els.exportEnd && !els.exportEnd.value) els.exportEnd.value = formatDateInput(range.end);
    if (els.exportSubmit) {
      els.exportSubmit.addEventListener('click', async () => {
        if (!authIsValid(state.auth)) {
          setStatus(els.exportStatus, 'Sign in to export applications.', 'error');
          return;
        }
        if (!config.apiBase) {
          setStatus(els.exportStatus, 'Set the API base URL to export applications.', 'error');
          return;
        }
        let startValue = (els.exportStart?.value || '').trim();
        let endValue = (els.exportEnd?.value || '').trim();
        if (!startValue && !endValue) {
          const fallback = defaultRange();
          startValue = formatDateInput(fallback.start);
          endValue = formatDateInput(fallback.end);
          if (els.exportStart) els.exportStart.value = startValue;
          if (els.exportEnd) els.exportEnd.value = endValue;
        }
        if (!startValue || !endValue) {
          setStatus(els.exportStatus, 'Choose a start and end date for the export.', 'error');
          return;
        }
        const startDate = parseDateInput(startValue);
        const endDate = parseDateInput(endValue);
        if (!startDate || !endDate) {
          setStatus(els.exportStatus, 'Export dates must be valid.', 'error');
          return;
        }
        let start = startValue;
        let end = endValue;
        if (startDate > endDate) {
          start = endValue;
          end = startValue;
          if (els.exportStart) els.exportStart.value = start;
          if (els.exportEnd) els.exportEnd.value = end;
        }
        try {
          setStatus(els.exportStatus, 'Preparing export...', 'info');
          const result = await requestJson('/api/exports', {
            method: 'POST',
            body: { start, end }
          });
          const url = result?.downloadUrl;
          if (url) {
            const link = document.createElement('a');
            link.href = url;
            link.download = '';
            document.body.appendChild(link);
            link.click();
            link.remove();
            setStatus(els.exportStatus, 'Export ready. Download should begin shortly.', 'success');
          } else {
            setStatus(els.exportStatus, 'Export ready, but no download link was returned.', 'error');
          }
        } catch (err) {
          console.error('Export failed', err);
          setStatus(els.exportStatus, err?.message || 'Unable to export applications.', 'error');
        }
      });
    }
  };

  const updateEntryStatus = async (entry, nextStatus, statusDate = '') => {
    if (!entry || !entry.applicationId) return false;
    const statusTarget = els.detailModalStatus || els.entryListStatus;
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to update statuses.', 'error');
      return false;
    }
    if (!config.apiBase) {
      setStatus(statusTarget, 'Set the API base URL to update statuses.', 'error');
      return false;
    }
    const entryType = entry.entryType || getEntryType(entry);
    const normalizedStatus = (nextStatus || '').toString().trim();
    if (!normalizedStatus) {
      setStatus(statusTarget, 'Choose a status to apply.', 'error');
      return false;
    }
    const payload = { status: normalizedStatus };
    let safeDate = (statusDate || '').toString().trim();
    if (safeDate && !parseDateInput(safeDate)) {
      setStatus(statusTarget, 'Status date must be valid (YYYY-MM-DD).', 'error');
      return false;
    }
    if (!safeDate && entryType === 'application') {
      safeDate = formatDateInput(new Date());
    }
    if (safeDate) {
      payload.statusDate = safeDate;
    }
    try {
      setStatus(statusTarget, 'Updating status...', 'info');
      const endpoint = entryType === 'prospect'
        ? `/api/prospects/${encodeURIComponent(entry.applicationId)}`
        : `/api/applications/${encodeURIComponent(entry.applicationId)}`;
      await requestJson(endpoint, { method: 'PATCH', body: payload });
      const nextHistory = Array.isArray(entry.statusHistory) ? [...entry.statusHistory] : [];
      nextHistory.push({
        status: normalizedStatus,
        date: payload.statusDate || new Date().toISOString()
      });
      const updatedEntry = normalizeEntry({
        ...entry,
        status: normalizedStatus,
        statusDate: payload.statusDate || entry.statusDate,
        statusHistory: nextHistory
      }, entryType);
      if (updatedEntry.applicationId) state.entryItems.set(updatedEntry.applicationId, updatedEntry);
      renderDetailModal(updatedEntry);
      setStatus(statusTarget, 'Status updated.', 'success');
      await Promise.all([refreshEntries(), refreshDashboard()]);
      return true;
    } catch (err) {
      console.error('Status update failed', err);
      setStatus(statusTarget, err?.message || 'Unable to update status.', 'error');
      return false;
    }
  };

  const hideUndoBanner = () => {
    if (!els.entryUndoBanner) return;
    els.entryUndoBanner.dataset.state = 'hidden';
    state.lastDeletedEntry = null;
    if (state.entryUndoTimer) {
      window.clearTimeout(state.entryUndoTimer);
      state.entryUndoTimer = null;
    }
  };

  const showUndoBanner = (entry) => {
    if (!els.entryUndoBanner || !els.entryUndoMessage || !els.entryUndoAction) return;
    if (!entry) return;
    state.lastDeletedEntry = entry;
    const label = getEntryLabel(entry);
    const hasAttachments = Array.isArray(entry.attachments) && entry.attachments.length;
    els.entryUndoMessage.textContent = hasAttachments
      ? `${label} deleted. Undo restores details (attachments are not restored).`
      : `${label} deleted. Undo?`;
    els.entryUndoBanner.dataset.state = '';
    if (state.entryUndoTimer) window.clearTimeout(state.entryUndoTimer);
    state.entryUndoTimer = window.setTimeout(() => hideUndoBanner(), 10000);
  };

  const restoreDeletedEntry = async () => {
    const entry = state.lastDeletedEntry;
    if (!entry) return;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to restore entries.', 'error');
      return;
    }
    if (!config.apiBase) {
      setStatus(els.entryListStatus, 'Set the API base URL to restore entries.', 'error');
      return;
    }
    const entryType = entry.entryType || getEntryType(entry);
    const company = (entry.company || '').toString().trim();
    const title = (entry.title || '').toString().trim();
    const payload = {
      company,
      title,
      status: getEntryStatusLabel(entry),
      notes: (entry.notes || '').toString().trim()
    };
    if (entry.location) payload.location = entry.location;
    if (entry.source) payload.source = entry.source;
    if (entry.batch) payload.batch = entry.batch;
    if (Array.isArray(entry.tags) && entry.tags.length) payload.tags = entry.tags;
    if (entry.followUpDate) payload.followUpDate = entry.followUpDate;
    if (entry.followUpNote) payload.followUpNote = entry.followUpNote;
    if (entry.customFields && typeof entry.customFields === 'object' && Object.keys(entry.customFields).length) {
      payload.customFields = entry.customFields;
    }
    if (entry.postingDate) payload.postingDate = entry.postingDate;
    if (entry.captureDate) payload.captureDate = entry.captureDate;
    if (entry.jobUrl) payload.jobUrl = entry.jobUrl;
    if (entryType === 'application') {
      const appliedDate = (entry.appliedDate || '').toString().trim();
      if (!company || !title || !appliedDate) {
        setStatus(els.entryListStatus, 'Unable to restore. Missing company, title, or applied date.', 'error');
        return;
      }
      payload.appliedDate = appliedDate;
    } else {
      const jobUrl = (entry.jobUrl || '').toString().trim();
      if (!company || !title || !jobUrl) {
        setStatus(els.entryListStatus, 'Unable to restore. Missing company, title, or job URL.', 'error');
        return;
      }
      if (entry.captureDate) payload.captureDate = entry.captureDate;
    }
    try {
      setStatus(els.entryListStatus, 'Restoring entry...', 'info');
      const endpoint = entryType === 'prospect' ? '/api/prospects' : '/api/applications';
      await requestJson(endpoint, { method: 'POST', body: payload });
      hideUndoBanner();
      setStatus(els.entryListStatus, 'Entry restored.', 'success');
      await Promise.all([refreshEntries(), refreshDashboard()]);
    } catch (err) {
      console.error('Entry restore failed', err);
      setStatus(els.entryListStatus, err?.message || 'Unable to restore entry.', 'error');
    }
  };

  const deleteEntry = async (entryId) => {
    if (!entryId) return;
    if (!config.apiBase) {
      setStatus(els.entryListStatus, 'Set the API base URL to delete entries.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to delete entries.', 'error');
      return;
    }
    const item = state.entryItems.get(entryId);
    const label = [item?.title, item?.company].filter(Boolean).join(' · ') || 'this entry';
    if (!confirmAction(`Delete ${label}? This cannot be undone.`)) return;
    try {
      setStatus(els.entryListStatus, 'Deleting entry...', 'info');
      await requestJson(`/api/applications/${encodeURIComponent(entryId)}`, { method: 'DELETE' });
      if (state.editingEntryId === entryId) {
        clearEntryEditMode('Ready to save entries.', 'info');
        if (els.entryForm) {
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          resetEntryDateFields();
          clearAttachmentInputs();
        }
      }
      if (state.selectedEntryIds.has(entryId)) {
        state.selectedEntryIds.delete(entryId);
        updateEntrySelectionUI();
      }
      setStatus(els.entryListStatus, 'Entry deleted.', 'success');
      showUndoBanner(item);
      await Promise.all([refreshEntries(), refreshDashboard()]);
    } catch (err) {
      console.error('Entry delete failed', err);
      setStatus(els.entryListStatus, err?.message || 'Unable to delete entry.', 'error');
    }
  };

  const downloadAttachment = async (attachment, statusEl = els.detailModalStatus || els.entryListStatus) => {
    if (!attachment) return;
    if (!authIsValid(state.auth)) {
      setStatus(statusEl, 'Sign in to download attachments.', 'error');
      return;
    }
    if (!config.apiBase) {
      setStatus(statusEl, 'Set the API base URL to download attachments.', 'error');
      return;
    }
    const key = (attachment.key || '').toString().trim();
    if (!key) {
      setStatus(statusEl, 'Attachment key missing.', 'error');
      return;
    }
    const filename = (attachment.filename || attachment.name || 'attachment').toString().trim();
    try {
      setStatus(statusEl, `Preparing ${filename}...`, 'info');
      const data = await requestJson('/api/attachments/download', {
        method: 'POST',
        body: { key }
      });
      const url = data?.downloadUrl;
      if (!url) {
        setStatus(statusEl, 'Download link unavailable.', 'error');
        return;
      }
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || '';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus(statusEl, 'Download started.', 'success');
    } catch (err) {
      console.error('Attachment download failed', err);
      setStatus(statusEl, err?.message || 'Unable to download attachment.', 'error');
    }
  };

  const downloadEntryZip = async (entryId, statusEl = els.entryListStatus) => {
    if (!entryId) return;
    if (!authIsValid(state.auth)) {
      setStatus(statusEl, 'Sign in to download attachments.', 'error');
      return;
    }
    try {
      const item = state.entryItems.get(entryId);
      const label = [item?.title, item?.company].filter(Boolean).join(' · ') || 'entry';
      setStatus(statusEl, `Preparing ${label} attachments...`, 'info');
      const data = await requestAttachmentZip(entryId);
      const url = data?.downloadUrl;
      if (!url) {
        setStatus(statusEl, 'Download link unavailable.', 'error');
        return;
      }
      const link = document.createElement('a');
      link.href = url;
      link.download = '';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus(statusEl, 'Download started.', 'success');
    } catch (err) {
      console.error('Download zip failed', err);
      setStatus(statusEl, err?.message || 'Unable to download attachments zip.', 'error');
    }
  };

  const deleteEntriesBulk = async (entryIds = []) => {
    const ids = entryIds.filter(Boolean);
    if (!ids.length) return;
    if (!config.apiBase) {
      setStatus(els.entryListStatus, 'Set the API base URL to delete entries.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to delete entries.', 'error');
      return;
    }
    const label = ids.length === 1 ? 'this entry' : `${ids.length} entries`;
    if (!confirmAction(`Delete ${label}? This cannot be undone.`)) return;
    try {
      setStatus(els.entryListStatus, `Deleting ${label}...`, 'info');
      const results = await runWithConcurrency(ids, 3, async (entryId) => {
        await requestJson(`/api/applications/${encodeURIComponent(entryId)}`, { method: 'DELETE' });
        return entryId;
      });
      const failed = results.filter(result => !result.ok).length;
      const deleted = ids.length - failed;
      if (state.editingEntryId && ids.includes(state.editingEntryId)) {
        clearEntryEditMode('Ready to save entries.', 'info');
        if (els.entryForm) {
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          resetEntryDateFields();
          clearAttachmentInputs();
        }
      }
      state.selectedEntryIds.clear();
      updateEntrySelectionUI();
      hideUndoBanner();
      await Promise.all([refreshEntries(), refreshDashboard()]);
      if (failed) {
        setStatus(els.entryListStatus, `${deleted} deleted, ${failed} failed.`, 'error');
      } else {
        setStatus(els.entryListStatus, `Deleted ${deleted} entries.`, 'success');
      }
    } catch (err) {
      console.error('Bulk delete failed', err);
      setStatus(els.entryListStatus, err?.message || 'Unable to delete entries.', 'error');
    }
  };

  const applyBulkStatus = async () => {
    const status = (els.bulkStatusSelect?.value || '').toString().trim();
    if (!status) {
      setStatus(els.entryListStatus, 'Choose a status to apply.', 'error');
      return;
    }
    if (!config.apiBase) {
      setStatus(els.entryListStatus, 'Set the API base URL to update statuses.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(els.entryListStatus, 'Sign in to update statuses.', 'error');
      return;
    }
    const selectedIds = Array.from(state.selectedEntryIds);
    const applicationIds = selectedIds.filter((id) => {
      const item = state.entryItems.get(id);
      return item?.entryType === 'application';
    });
    const skippedProspects = selectedIds.length - applicationIds.length;
    if (!applicationIds.length) {
      setStatus(els.entryListStatus, 'Select at least one application to update.', 'error');
      return;
    }
    let statusDate = (els.bulkStatusDate?.value || '').toString().trim();
    if (statusDate && !parseDateInput(statusDate)) {
      setStatus(els.entryListStatus, 'Status date must be valid (YYYY-MM-DD).', 'error');
      return;
    }
    if (!statusDate) {
      statusDate = formatDateInput(new Date());
      if (els.bulkStatusDate) els.bulkStatusDate.value = statusDate;
    }
    const label = applicationIds.length === 1 ? '1 application' : `${applicationIds.length} applications`;
    try {
      setStatus(els.entryListStatus, `Updating ${label}...`, 'info');
      const results = await runWithConcurrency(applicationIds, 3, async (entryId) => {
        await requestJson(`/api/applications/${encodeURIComponent(entryId)}`, {
          method: 'PATCH',
          body: { status, statusDate }
        });
        return entryId;
      });
      const failed = results.filter(result => !result.ok).length;
      const updated = applicationIds.length - failed;
      const parts = [
        `Updated ${updated} ${updated === 1 ? 'application' : 'applications'} to ${status}.`
      ];
      if (failed) parts.push(`${failed} failed.`);
      if (skippedProspects) {
        parts.push(`Skipped ${skippedProspects} prospect${skippedProspects === 1 ? '' : 's'}.`);
      }
      setStatus(els.entryListStatus, parts.join(' '), failed ? 'error' : 'success');
      if (updated) {
        await Promise.all([refreshEntries(), refreshDashboard()]);
      }
    } catch (err) {
      console.error('Bulk status update failed', err);
      setStatus(els.entryListStatus, err?.message || 'Unable to update statuses.', 'error');
    }
  };

  const submitProspect = async (payload) => {
    if (!els.entryFormStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryFormStatus, 'Sign in to save prospects.', 'error');
      return false;
    }
    const editingId = state.editingEntryId;
    try {
      setStatus(els.entryFormStatus, editingId ? 'Updating prospect...' : 'Saving prospect...', 'info');
      if (editingId) {
        await requestJson(`/api/prospects/${encodeURIComponent(editingId)}`, { method: 'PATCH', body: payload });
      } else {
        await requestJson('/api/prospects', { method: 'POST', body: payload });
      }
      clearEntryEditMode(editingId ? 'Prospect updated.' : 'Prospect saved.', 'success');
      clearEntryDraft();
      setDraftStatus('Saved to tracker.');
      await sleep(200);
      await refreshEntries();
      return true;
    } catch (err) {
      console.error('Prospect save failed', err);
      setStatus(els.entryFormStatus, err?.message || 'Unable to save prospect.', 'error');
      return false;
    }
  };

  const buildApplicationPayloadFromProspect = (item, appliedDate) => {
    const payload = {
      company: (item?.company || '').toString().trim(),
      title: (item?.title || '').toString().trim(),
      appliedDate,
      status: 'Applied',
      statusDate: appliedDate,
      notes: (item?.notes || '').toString().trim()
    };
    if (item?.jobUrl) payload.jobUrl = item.jobUrl;
    if (item?.location) payload.location = item.location;
    if (item?.source) payload.source = item.source;
    if (item?.batch) payload.batch = item.batch;
    if (item?.postingDate) payload.postingDate = item.postingDate;
    if (item?.captureDate) payload.captureDate = item.captureDate;
    if (Array.isArray(item?.tags) && item.tags.length) payload.tags = item.tags;
    if (item?.followUpDate) payload.followUpDate = item.followUpDate;
    if (item?.followUpNote) payload.followUpNote = item.followUpNote;
    if (item?.customFields && typeof item.customFields === 'object' && Object.keys(item.customFields).length) {
      payload.customFields = item.customFields;
    }
    return payload;
  };

  const convertProspectToApplication = async (payload, prospectId, statusEl = null) => {
    const statusTarget = statusEl || els.entryListStatus || els.entryFormStatus;
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to save applications.', 'error');
      return false;
    }
    try {
      setStatus(statusTarget, 'Moving prospect to applications...', 'info');
      await requestJson('/api/applications', { method: 'POST', body: payload });
      let deleteError = null;
      if (prospectId) {
        try {
          await requestJson(`/api/applications/${encodeURIComponent(prospectId)}`, { method: 'DELETE' });
        } catch (err) {
          deleteError = err;
        }
      }
      if (deleteError) {
        setStatus(statusTarget, 'Application saved, but the prospect could not be removed.', 'error');
      } else {
        setStatus(statusTarget, prospectId ? 'Prospect moved to applications.' : 'Application saved.', 'success');
      }
      clearEntryEditMode();
      await sleep(200);
      await Promise.all([refreshEntries(), refreshDashboard()]);
      return true;
    } catch (err) {
      console.error('Prospect conversion failed', err);
      setStatus(statusTarget, err?.message || 'Unable to move prospect to applications.', 'error');
      return false;
    }
  };

  const convertProspectToApplicationWithAttachments = async (entry, appliedDate, attachments = [], statusEl = null) => {
    const statusTarget = statusEl || els.prospectReviewStatus || els.entryListStatus || els.entryFormStatus;
    if (!entry || !entry.applicationId) return false;
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to save applications.', 'error');
      return false;
    }
    if (!config.apiBase) {
      setStatus(statusTarget, 'Set the API base URL to save applications.', 'error');
      return false;
    }
    const payload = buildApplicationPayloadFromProspect(entry, appliedDate);
    if (attachments.length && !validateAttachmentFiles(attachments, statusTarget)) {
      return false;
    }
    try {
      setStatus(statusTarget, 'Saving application...', 'info');
      const created = await requestJson('/api/applications', { method: 'POST', body: payload });
      const applicationId = created?.applicationId;
      if (!applicationId) {
        setStatus(statusTarget, 'Application saved, but no ID was returned.', 'error');
        return false;
      }
      let attachmentError = null;
      if (attachments.length && applicationId) {
        try {
          const label = attachments.length === 1 ? 'attachment' : 'attachments';
          setStatus(statusTarget, `Uploading ${attachments.length} ${label}...`, 'info');
          const uploaded = await uploadAttachments(applicationId, attachments, null, statusTarget);
          await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, {
            method: 'PATCH',
            body: { attachments: uploaded }
          });
        } catch (err) {
          attachmentError = err;
        }
      }
      let deleteError = null;
      try {
        await requestJson(`/api/applications/${encodeURIComponent(entry.applicationId)}`, { method: 'DELETE' });
      } catch (err) {
        deleteError = err;
      }
      const messages = [];
      if (attachmentError) {
        messages.push('Application saved, but attachments failed to upload.');
      }
      if (deleteError) {
        messages.push('Prospect could not be removed.');
      }
      if (!messages.length) {
        messages.push('Prospect moved to applications.');
      }
      setStatus(statusTarget, messages.join(' '), attachmentError || deleteError ? 'error' : 'success');
      clearEntryEditMode();
      await sleep(200);
      await Promise.all([refreshEntries(), refreshDashboard()]);
      return true;
    } catch (err) {
      console.error('Prospect conversion failed', err);
      setStatus(statusTarget, err?.message || 'Unable to move prospect to applications.', 'error');
      return false;
    }
  };

  const archiveProspectEntry = async (entry, statusEl = null) => {
    if (!entry || !entry.applicationId) return;
    const statusTarget = statusEl || els.prospectReviewStatus || els.entryListStatus;
    if (!config.apiBase) {
      setStatus(statusTarget, 'Set the API base URL to archive prospects.', 'error');
      return;
    }
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to archive prospects.', 'error');
      return;
    }
    const label = [entry.title, entry.company].filter(Boolean).join(' · ') || 'this prospect';
    if (!confirmAction(`Reject and archive ${label}? It will move to your rejected list.`)) return;
    const statusDate = formatDateInput(new Date());
    const rejectNote = promptActionNote('Optional: add a quick note on why you rejected this role (leave blank to skip).', '');
    try {
      setStatus(statusTarget, 'Archiving prospect...', 'info');
      await requestJson(`/api/prospects/${encodeURIComponent(entry.applicationId)}`, {
        method: 'PATCH',
        body: {
          status: 'Rejected',
          statusDate,
          ...(rejectNote ? { notes: appendEntryNote(entry.notes, `Rejection note (${statusDate}): ${rejectNote}`) } : {})
        }
      });
      setStatus(statusTarget, 'Prospect rejected.', 'success');
      await Promise.all([refreshEntries(), refreshDashboard()]);
    } catch (err) {
      console.error('Prospect archive failed', err);
      setStatus(statusTarget, err?.message || 'Unable to archive prospect.', 'error');
    }
  };

  const applyProspect = async (prospectId, appliedDateValue = '', statusEl = null) => {
    if (!prospectId) return false;
    const item = state.entryItems.get(prospectId);
    if (!item) return false;
    const statusTarget = statusEl || els.entryListStatus;
    if (!authIsValid(state.auth)) {
      setStatus(statusTarget, 'Sign in to move prospects.', 'error');
      return false;
    }
    const captureDate = item.captureDate && parseDateInput(item.captureDate) ? item.captureDate : '';
    const defaultDate = captureDate || formatDateInput(new Date());
    let appliedDate = (appliedDateValue || '').toString().trim();
    if (appliedDate && !parseDateInput(appliedDate)) {
      setStatus(statusTarget, 'Applied date must be valid (YYYY-MM-DD).', 'error');
      return false;
    }
    if (!appliedDate) {
      appliedDate = defaultDate;
    }
    const payload = buildApplicationPayloadFromProspect(item, appliedDate);
    return convertProspectToApplication(payload, prospectId, statusTarget);
  };

  const submitApplication = async (payload, attachments = []) => {
    if (!els.entryFormStatus) return false;
    if (!authIsValid(state.auth)) {
      setStatus(els.entryFormStatus, 'Sign in to save applications.', 'error');
      return false;
    }
    const editingId = state.editingEntryId;
    const editingItem = state.editingEntry;
    try {
      setStatus(els.entryFormStatus, editingId ? 'Updating application...' : 'Saving application...', 'info');
      let applicationId = editingId;
      let currentAttachments = Array.isArray(editingItem?.attachments) ? editingItem.attachments : [];
      if (editingId) {
        await requestJson(`/api/applications/${encodeURIComponent(editingId)}`, { method: 'PATCH', body: payload });
      } else {
        const created = await requestJson('/api/applications', { method: 'POST', body: payload });
        applicationId = created?.applicationId;
        currentAttachments = Array.isArray(created?.attachments) ? created.attachments : [];
      }
      let attachmentError = null;
      if (attachments.length && applicationId) {
        try {
          const label = attachments.length === 1 ? 'attachment' : 'attachments';
          setStatus(els.entryFormStatus, `Uploading ${attachments.length} ${label}...`, 'info');
          const uploaded = await uploadAttachments(applicationId, attachments, null, els.entryFormStatus);
          const maxCount = config.maxAttachmentCount || 12;
          const merged = [...currentAttachments, ...uploaded].slice(-maxCount);
          await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, {
            method: 'PATCH',
            body: { attachments: merged }
          });
        } catch (err) {
          attachmentError = err;
        }
      }
      if (attachmentError) {
        console.error('Attachment upload failed', attachmentError);
        clearEntryEditMode(
          editingId ? 'Updated application, but attachments failed to upload.' : 'Saved application, but attachments failed to upload.',
          'error'
        );
      } else {
        clearEntryEditMode(
          editingId ? 'Application updated. Refreshing dashboards...' : 'Application saved. Updating dashboards...',
          'success'
        );
      }
      clearEntryDraft();
      setDraftStatus('Saved to tracker.');
      await sleep(200);
      await Promise.all([refreshDashboard(), refreshEntries()]);
      return true;
    } catch (err) {
      console.error('Application save failed', err);
      setStatus(els.entryFormStatus, err?.message || 'Unable to save application.', 'error');
      return false;
    }
  };

  const initEntryForm = () => {
    if (!els.entryForm) return;
    updateAttachmentLimitText();
    if (els.customFieldAdd) {
      els.customFieldAdd.addEventListener('click', () => {
        addCustomFieldRow();
        scheduleEntryDraftSave();
      });
    }
    initUnknownDateToggle(els.postingDateInput, els.postingUnknownInput, true);
    setEntryType(state.entryType, { preserveStatus: false });
    resetEntryDateFields(state.entryType);
    restoreEntryDraft();
    if (els.resumePromptDownload) {
      els.resumePromptDownload.addEventListener('click', () => {
        const context = buildPromptContextFromForm();
        downloadEntryPrompt(context, 'resume');
      });
    }
    if (els.coverPromptDownload) {
      els.coverPromptDownload.addEventListener('click', () => {
        const context = buildPromptContextFromForm();
        downloadEntryPrompt(context, 'cover');
      });
    }
    if (els.entryTypeInputs.length) {
      els.entryTypeInputs.forEach((input) => {
        input.addEventListener('change', () => {
          const nextType = input.value;
          if (state.editingEntry && getEntryType(state.editingEntry) !== nextType) {
            clearEntryEditMode('Entry type changed. Start a new entry below.', 'info');
          }
          setEntryType(nextType);
          resetEntryDateFields(nextType);
        });
      });
    }
    if (els.jobUrlInput) {
      els.jobUrlInput.addEventListener('blur', () => maybeAutofillFromJobUrl());
      els.jobUrlInput.addEventListener('change', () => maybeAutofillFromJobUrl());
    }
    const handleDraftInput = () => {
      if (state.isResettingEntry) return;
      scheduleEntryDraftSave();
    };
    els.entryForm.addEventListener('input', handleDraftInput);
    els.entryForm.addEventListener('change', handleDraftInput);
    els.entryForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(els.entryForm);
      const entryType = (formData.get('entryType') || state.entryType).toString().trim();
      const company = (formData.get('company') || '').toString().trim();
      const title = (formData.get('title') || '').toString().trim();
      const jobUrl = normalizeUrl(formData.get('jobUrl'));
      const location = (formData.get('location') || '').toString().trim();
      const source = (formData.get('source') || '').toString().trim();
      const batch = (formData.get('batch') || '').toString().trim();
      const appliedDate = (formData.get('appliedDate') || '').toString().trim();
      const postingDate = (formData.get('postingDate') || '').toString().trim();
      const postingUnknown = Boolean(formData.get('postingDateUnknown'));
      const captureDate = (formData.get('captureDate') || '').toString().trim();
      const status = (formData.get('status') || (entryType === 'prospect' ? 'Active' : 'Applied')).toString().trim();
      const notes = (formData.get('notes') || '').toString().trim();
      const tags = parseTagInput(formData.get('tags') || '');
      const followUpDate = (formData.get('followUpDate') || '').toString().trim();
      const followUpNote = (formData.get('followUpNote') || '').toString().trim();
      const customFields = readCustomFields();
      const editing = state.editingEntry;

      if (!company || !title) {
        setStatus(els.entryFormStatus, 'Company and role title are required.', 'error');
        return;
      }
      if (!postingUnknown && !postingDate) {
        setStatus(els.entryFormStatus, 'Add a posting date or mark it as unknown.', 'error');
        return;
      }
      if (postingDate && !parseDateInput(postingDate)) {
        setStatus(els.entryFormStatus, 'Posting date must be valid.', 'error');
        return;
      }
      if (followUpDate && !parseDateInput(followUpDate)) {
        setStatus(els.entryFormStatus, 'Follow-up date must be valid.', 'error');
        return;
      }

      if (entryType === 'application') {
        if (!appliedDate) {
          setStatus(els.entryFormStatus, 'Applied date is required for applications.', 'error');
          return;
        }
        if (!parseDateInput(appliedDate)) {
          setStatus(els.entryFormStatus, 'Applied date must be valid.', 'error');
          return;
        }
        if (captureDate && !parseDateInput(captureDate)) {
          setStatus(els.entryFormStatus, 'Found date must be valid.', 'error');
          return;
        }
        const attachments = collectAttachments();
        if (!validateAttachmentFiles(attachments, els.entryFormStatus)) return;
        const payload = { company, title, appliedDate, notes };
        if (jobUrl || editing?.jobUrl) payload.jobUrl = jobUrl;
        if (location || editing?.location) payload.location = location;
        if (source || editing?.source) payload.source = source;
        if (batch || editing?.batch) payload.batch = batch;
        if (tags.length || (editing?.tags && editing.tags.length)) payload.tags = tags;
        if (followUpDate) {
          payload.followUpDate = followUpDate;
        } else if (editing?.followUpDate) {
          payload.followUpDate = null;
        }
        if (followUpNote) {
          payload.followUpNote = followUpNote;
        } else if (editing?.followUpNote) {
          payload.followUpNote = '';
        }
        if (Object.keys(customFields).length || (editing?.customFields && Object.keys(editing.customFields).length)) {
          payload.customFields = customFields;
        }
        if (postingUnknown) {
          payload.postingDate = null;
        } else if (postingDate) {
          payload.postingDate = postingDate;
        }
        if (captureDate) {
          payload.captureDate = captureDate;
        } else if (editing?.captureDate) {
          payload.captureDate = null;
        }
        const existingStatus = editing?.status ? editing.status.toString().trim().toLowerCase() : '';
        if (!editing || !existingStatus || status.toLowerCase() !== existingStatus) {
          payload.status = status;
        }
        const ok = await submitApplication(payload, attachments);
        if (ok) {
          const nextType = entryType;
          state.isResettingEntry = true;
          els.entryForm.reset();
          state.isResettingEntry = false;
          clearCustomFields();
          clearAttachmentInputs();
          setEntryType(nextType);
          resetEntryDateFields(nextType);
        }
        return;
      }

      if (!jobUrl) {
        setStatus(els.entryFormStatus, 'Job URL is required for prospects.', 'error');
        return;
      }
      if (!captureDate) {
        setStatus(els.entryFormStatus, 'Found date is required for prospects.', 'error');
        return;
      }
      if (!parseDateInput(captureDate)) {
        setStatus(els.entryFormStatus, 'Found date must be valid.', 'error');
        return;
      }
      const payload = { company, title, jobUrl, location, source, status, notes, captureDate };
      if (batch || editing?.batch) payload.batch = batch;
      if (tags.length || (editing?.tags && editing.tags.length)) payload.tags = tags;
      if (followUpDate) {
        payload.followUpDate = followUpDate;
      } else if (editing?.followUpDate) {
        payload.followUpDate = null;
      }
      if (followUpNote) {
        payload.followUpNote = followUpNote;
      } else if (editing?.followUpNote) {
        payload.followUpNote = '';
      }
      if (Object.keys(customFields).length || (editing?.customFields && Object.keys(editing.customFields).length)) {
        payload.customFields = customFields;
      }
      if (postingUnknown) {
        payload.postingDate = null;
      } else if (postingDate) {
        payload.postingDate = postingDate;
      }
      const ok = await submitProspect(payload);
      if (ok) {
        const nextType = entryType;
        state.isResettingEntry = true;
        els.entryForm.reset();
        state.isResettingEntry = false;
        clearCustomFields();
        setEntryType(nextType);
        resetEntryDateFields(nextType);
      }
    });
    els.entryForm.addEventListener('reset', () => {
      clearAttachmentInputs();
      clearCustomFields();
      const nextType = state.entryType;
      resetEntryDateFields(nextType);
      if (state.isResettingEntry) return;
      setEntryType(nextType);
      clearEntryEditMode();
      clearEntryDraft();
    });
  };

  const parseImportPayloads = (text) => {
    const rows = parseCsv(text || '');
    if (!rows.length) return { entries: [], skipped: 0, missing: ['company', 'title', 'appliedDate'] };
    const headers = rows.shift().map(header => header.trim());
    const map = buildHeaderMap(headers);
    const missing = ['company', 'title', 'appliedDate'].filter(key => map[key] === undefined);
    if (missing.length) return { entries: [], skipped: rows.length, missing };

    const entries = [];
    let skipped = 0;
    rows.forEach((row) => {
      const company = (row[map.company] || '').toString().trim();
      const title = (row[map.title] || '').toString().trim();
      const appliedDate = parseCsvDate(row[map.appliedDate]);
      const postingDate = map.postingDate !== undefined ? parseCsvDate(row[map.postingDate]) : '';
      const status = map.status !== undefined ? (row[map.status] || '').toString().trim() : '';
      const captureDate = map.captureDate !== undefined ? parseCsvDate(row[map.captureDate]) : '';
      const notes = map.notes !== undefined ? (row[map.notes] || '').toString().trim() : '';
      const tagsValue = map.tags !== undefined ? (row[map.tags] || '').toString().trim() : '';
      const followUpDate = map.followUpDate !== undefined ? parseCsvDate(row[map.followUpDate]) : '';
      const followUpNote = map.followUpNote !== undefined ? (row[map.followUpNote] || '').toString().trim() : '';
      const customFieldsValue = map.customFields !== undefined ? (row[map.customFields] || '').toString().trim() : '';
      const jobUrl = map.jobUrl !== undefined ? normalizeUrl(row[map.jobUrl]) : '';
      const location = map.location !== undefined ? (row[map.location] || '').toString().trim() : '';
      const source = map.source !== undefined ? (row[map.source] || '').toString().trim() : '';
      const batch = map.batch !== undefined ? (row[map.batch] || '').toString().trim() : '';
      const attachmentsValue = map.attachments !== undefined ? (row[map.attachments] || '').toString().trim() : '';
      const resumeFile = map.resumeFile !== undefined ? (row[map.resumeFile] || '').toString().trim() : '';
      const coverLetterFile = map.coverLetterFile !== undefined ? (row[map.coverLetterFile] || '').toString().trim() : '';
      if (!company || !title || !appliedDate) {
        skipped += 1;
        return;
      }
      const payload = {
        company,
        title,
        appliedDate,
        status: status || 'Applied',
        notes
      };
      if (batch) payload.batch = batch;
      if (postingDate) payload.postingDate = postingDate;
      if (jobUrl) payload.jobUrl = jobUrl;
      if (location) payload.location = location;
      if (source) payload.source = source;
      if (captureDate) payload.captureDate = captureDate;
      const tags = parseTagList(tagsValue);
      if (tags.length) payload.tags = tags;
      if (followUpDate) payload.followUpDate = followUpDate;
      if (followUpNote) payload.followUpNote = followUpNote;
      const customFields = parseCustomFieldsInput(customFieldsValue);
      if (Object.keys(customFields).length) payload.customFields = customFields;
      const attachmentFiles = [];
      const attachmentLookup = new Set();
      const addAttachment = (name, kind) => {
        if (!name) return;
        const key = name.toLowerCase();
        if (attachmentLookup.has(key)) return;
        attachmentLookup.add(key);
        attachmentFiles.push({ name, kind });
      };
      parseAttachmentList(attachmentsValue).forEach(name => addAttachment(name, 'attachment'));
      addAttachment(resumeFile, 'resume');
      addAttachment(coverLetterFile, 'cover-letter');
      entries.push({
        payload,
        attachmentFiles
      });
    });
    return { entries, skipped, missing: [] };
  };

  const parseProspectPayloads = (text) => {
    const rows = parseCsv(text || '');
    if (!rows.length) return { entries: [], skipped: 0, missing: ['company', 'title', 'jobUrl', 'captureDate'] };
    const headers = rows.shift().map(header => header.trim());
    const map = buildHeaderMap(headers);
    const missing = ['company', 'title', 'jobUrl', 'captureDate'].filter(key => map[key] === undefined);
    if (missing.length) return { entries: [], skipped: rows.length, missing };

    const entries = [];
    let skipped = 0;
    rows.forEach((row) => {
      const company = (row[map.company] || '').toString().trim();
      const title = (row[map.title] || '').toString().trim();
      const jobUrl = map.jobUrl !== undefined ? normalizeUrl(row[map.jobUrl]) : '';
      const location = map.location !== undefined ? (row[map.location] || '').toString().trim() : '';
      const source = map.source !== undefined ? (row[map.source] || '').toString().trim() : '';
      const batch = map.batch !== undefined ? (row[map.batch] || '').toString().trim() : '';
      const postingDate = map.postingDate !== undefined ? parseCsvDate(row[map.postingDate]) : '';
      const captureDate = map.captureDate !== undefined ? parseCsvDate(row[map.captureDate]) : '';
      const status = map.status !== undefined ? (row[map.status] || '').toString().trim() : '';
      const notes = map.notes !== undefined ? (row[map.notes] || '').toString().trim() : '';
      const tagsValue = map.tags !== undefined ? (row[map.tags] || '').toString().trim() : '';
      const followUpDate = map.followUpDate !== undefined ? parseCsvDate(row[map.followUpDate]) : '';
      const followUpNote = map.followUpNote !== undefined ? (row[map.followUpNote] || '').toString().trim() : '';
      const customFieldsValue = map.customFields !== undefined ? (row[map.customFields] || '').toString().trim() : '';
      if (!company || !title || !jobUrl || !captureDate) {
        skipped += 1;
        return;
      }
      const payload = {
        company,
        title,
        jobUrl,
        captureDate,
        status: status || 'Active',
        notes
      };
      if (batch) payload.batch = batch;
      if (postingDate) payload.postingDate = postingDate;
      if (location) payload.location = location;
      if (source) payload.source = source;
      const tags = parseTagList(tagsValue);
      if (tags.length) payload.tags = tags;
      if (followUpDate) payload.followUpDate = followUpDate;
      if (followUpNote) payload.followUpNote = followUpNote;
      const customFields = parseCustomFieldsInput(customFieldsValue);
      if (Object.keys(customFields).length) payload.customFields = customFields;
      entries.push({ payload });
    });
    return { entries, skipped, missing: [] };
  };

  const buildImportAttachmentMap = (files = []) => {
    const map = new Map();
    files.forEach((file) => {
      if (file && file.name) {
        map.set(file.name.toLowerCase(), file);
      }
    });
    return map;
  };

  const resolveImportAttachment = (name, lookup, missing) => {
    const trimmed = (name || '').toString().trim();
    if (!trimmed) return null;
    const file = lookup.get(trimmed.toLowerCase());
    if (!file && missing) missing.add(trimmed);
    return file || null;
  };

  const initImport = () => {
    if (els.importTemplate) {
      els.importTemplate.addEventListener('click', () => {
        const blob = new Blob([CSV_TEMPLATE], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'job-applications-template.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      });
    }
    if (els.importSubmit) {
      els.importSubmit.addEventListener('click', async () => {
        resetImportProgress();
        if (!authIsValid(state.auth)) {
          setStatus(els.importStatus, 'Sign in to import applications.', 'error');
          return;
        }
        if (!config.apiBase) {
          setStatus(els.importStatus, 'Set the API base URL to import applications.', 'error');
          return;
        }
        const file = els.importFile?.files?.[0];
        if (!file) {
          setStatus(els.importStatus, 'Choose a CSV file to import.', 'error');
          return;
        }
        try {
          setStatus(els.importStatus, 'Reading CSV...', 'info');
          const text = await readFileText(file);
          const { entries, skipped, missing } = parseImportPayloads(text);
          if (missing.length) {
            setStatus(els.importStatus, `Missing columns: ${missing.join(', ')}.`, 'error');
            return;
          }
          if (!entries.length) {
            setStatus(els.importStatus, 'No valid rows found in the CSV.', 'error');
            return;
          }
          const importBatch = (els.importBatch?.value || '').toString().trim();
          if (importBatch) {
            entries.forEach((entry) => {
              entry.payload.batch = importBatch;
            });
          }
          const attachmentLookup = buildImportAttachmentMap(Array.from(els.importAttachments?.files || []));
          const totalEntries = entries.length;
          const totalAttachments = entries.reduce((sum, entry) => {
            const files = Array.isArray(entry.attachmentFiles) ? entry.attachmentFiles : [];
            const matched = files.filter(item => attachmentLookup.has((item?.name || '').toLowerCase())).length;
            return sum + matched;
          }, 0);
          let entriesProcessed = 0;
          let attachmentsProcessed = 0;
          const updateProgress = () => {
            const label = totalAttachments
              ? `Uploading ${attachmentsProcessed}/${totalAttachments} attachments · ${entriesProcessed}/${totalEntries} entries`
              : `Importing ${entriesProcessed}/${totalEntries} applications`;
            setImportProgress(entriesProcessed, totalEntries, label);
          };
          updateProgress();
          const missingAttachments = new Set();
          const importLabel = totalAttachments
            ? `Importing ${entries.length} applications and ${totalAttachments} attachments...`
            : `Importing ${entries.length} applications...`;
          setStatus(els.importStatus, importLabel, 'info');
          const handleAttachmentProgress = () => {
            attachmentsProcessed += 1;
            updateProgress();
          };
          const results = await runWithConcurrency(entries, 3, async (entry) => {
            let attachmentsUploaded = 0;
            let attachmentsFailed = 0;
            try {
              const created = await requestJson('/api/applications', {
                method: 'POST',
                body: entry.payload
              });
              const applicationId = created?.applicationId;
              const attachments = [];
              const fileSpecs = Array.isArray(entry.attachmentFiles) ? entry.attachmentFiles : [];
              fileSpecs.forEach((item) => {
                const file = resolveImportAttachment(item?.name, attachmentLookup, missingAttachments);
                if (file) attachments.push({ file, kind: item?.kind || 'attachment' });
              });
              if (attachments.length && applicationId) {
                try {
                  const uploaded = await uploadAttachments(applicationId, attachments, handleAttachmentProgress, els.importStatus);
                  await requestJson(`/api/applications/${encodeURIComponent(applicationId)}`, {
                    method: 'PATCH',
                    body: { attachments: uploaded }
                  });
                  attachmentsUploaded = uploaded.length;
                } catch {
                  attachmentsFailed = attachments.length;
                }
              } else if (attachments.length && !applicationId) {
                attachmentsFailed = attachments.length;
              }
            } finally {
              entriesProcessed += 1;
              updateProgress();
            }
            return { attachmentsUploaded, attachmentsFailed };
          });
          const success = results.filter(result => result.ok).length;
          const failed = results.length - success;
          const attachmentTotals = results.reduce((acc, result) => {
            if (result.ok) {
              acc.uploaded += result.value?.attachmentsUploaded || 0;
              acc.failed += result.value?.attachmentsFailed || 0;
            }
            return acc;
          }, { uploaded: 0, failed: 0 });
          const parts = [`Imported ${success} of ${entries.length} applications.`];
          if (skipped) parts.push(`Skipped ${skipped} rows.`);
          if (failed) parts.push(`${failed} failed.`);
          if (missingAttachments.size) {
            const missingList = Array.from(missingAttachments);
            const preview = missingList.slice(0, 3);
            const suffix = missingList.length > preview.length ? ', ...' : '';
            parts.push(`Missing ${missingList.length} attachment file${missingList.length === 1 ? '' : 's'}: ${preview.join(', ')}${suffix}.`);
          }
          if (attachmentTotals.uploaded) {
            parts.push(`Uploaded ${attachmentTotals.uploaded} attachment${attachmentTotals.uploaded === 1 ? '' : 's'}.`);
          }
          if (attachmentTotals.failed) {
            parts.push(`Attachment uploads failed for ${attachmentTotals.failed} file${attachmentTotals.failed === 1 ? '' : 's'}.`);
          }
          const hasIssues = failed || attachmentTotals.failed || missingAttachments.size;
          setStatus(els.importStatus, parts.join(' '), hasIssues ? 'error' : 'success');
          if (totalEntries) {
            const finalLabel = totalAttachments
              ? `Uploaded ${attachmentsProcessed}/${totalAttachments} attachments · ${entriesProcessed}/${totalEntries} entries`
              : `Imported ${entriesProcessed}/${totalEntries} applications`;
            setImportProgress(entriesProcessed, totalEntries, finalLabel);
          }
          if (success) {
            await Promise.all([refreshDashboard(), refreshEntries()]);
          }
          if (els.importFile) els.importFile.value = '';
          if (els.importAttachments) els.importAttachments.value = '';
        } catch (err) {
          console.error('CSV import failed', err);
          setStatus(els.importStatus, err?.message || 'Unable to import applications.', 'error');
        }
      });
    }
  };

  const initProspectImport = () => {
    if (els.prospectImportTemplate) {
      els.prospectImportTemplate.addEventListener('click', () => {
        const blob = new Blob([PROSPECT_CSV_TEMPLATE], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'job-prospects-template.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      });
    }
    if (els.prospectPromptDownload) {
      els.prospectPromptDownload.addEventListener('click', () => {
        const blob = new Blob([PROSPECT_PROMPT_TEMPLATE], { type: 'text/plain;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'job-prospect-prompt.txt';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      });
    }
    if (els.prospectImportSubmit) {
      els.prospectImportSubmit.addEventListener('click', async () => {
        resetProspectImportProgress();
        if (!authIsValid(state.auth)) {
          setStatus(els.prospectImportStatus, 'Sign in to import prospects.', 'error');
          return;
        }
        if (!config.apiBase) {
          setStatus(els.prospectImportStatus, 'Set the API base URL to import prospects.', 'error');
          return;
        }
        const file = els.prospectImportFile?.files?.[0];
        if (!file) {
          setStatus(els.prospectImportStatus, 'Choose a CSV file to import.', 'error');
          return;
        }
        try {
          setStatus(els.prospectImportStatus, 'Reading CSV...', 'info');
          const text = await readFileText(file);
          const { entries, skipped, missing } = parseProspectPayloads(text);
          if (missing.length) {
            setStatus(els.prospectImportStatus, `Missing columns: ${missing.join(', ')}.`, 'error');
            return;
          }
          if (!entries.length) {
            setStatus(els.prospectImportStatus, 'No valid rows found in the CSV.', 'error');
            return;
          }
          const importBatch = (els.prospectImportBatch?.value || '').toString().trim();
          if (importBatch) {
            entries.forEach((entry) => {
              entry.payload.batch = importBatch;
            });
          }
          const totalEntries = entries.length;
          let entriesProcessed = 0;
          const updateProgress = () => {
            const label = `Imported ${entriesProcessed}/${totalEntries} prospects`;
            setProspectImportProgress(entriesProcessed, totalEntries, label);
          };
          updateProgress();
          setStatus(els.prospectImportStatus, `Importing ${totalEntries} prospects...`, 'info');
          const results = await runWithConcurrency(entries, 3, async (entry) => {
            try {
              await requestJson('/api/prospects', {
                method: 'POST',
                body: entry.payload
              });
              return true;
            } finally {
              entriesProcessed += 1;
              updateProgress();
            }
          });
          const success = results.filter(result => result.ok).length;
          const failed = results.length - success;
          const parts = [`Imported ${success} of ${entries.length} prospects.`];
          if (skipped) parts.push(`Skipped ${skipped} rows.`);
          if (failed) parts.push(`${failed} failed.`);
          const hasIssues = failed || skipped;
          setStatus(els.prospectImportStatus, parts.join(' '), hasIssues ? 'error' : 'success');
          if (totalEntries) {
            setProspectImportProgress(entriesProcessed, totalEntries, `Imported ${entriesProcessed}/${totalEntries} prospects`);
          }
          if (success) {
            await refreshEntries();
          }
          if (els.prospectImportFile) els.prospectImportFile.value = '';
        } catch (err) {
          console.error('Prospect CSV import failed', err);
          setStatus(els.prospectImportStatus, err?.message || 'Unable to import prospects.', 'error');
        }
      });
    }
  };

	  const initFilters = () => {
	    const range = defaultRange();
	    updateRangeInputs(range);
	    if (els.filterReset) {
	      els.filterReset.addEventListener('click', () => {
	        const next = defaultRange();
	        updateRangeInputs(next);
	        state.range = next;
	        if (state.dashboardView === 'prospects') {
	          updateProspectDashboard();
	          return;
	        }
	        refreshDashboard();
	      });
	    }
	    if (els.filterRefresh) {
	      els.filterRefresh.addEventListener('click', () => {
	        if (state.dashboardView === 'prospects') {
	          const next = readRange();
	          state.range = next;
	          updateRangeInputs(next);
	          updateProspectDashboard();
	          return;
	        }
	        refreshDashboard();
	      });
	    }
	  };

  const initDashboardInteractions = () => {
    if (els.mapRemote) {
      const showRemote = () => showMapDetail('REMOTE', 'Remote', 'remote');
      els.mapRemote.addEventListener('click', showRemote);
      els.mapRemote.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        showRemote();
      });
    }
    if (els.detailReset) {
      els.detailReset.addEventListener('click', () => clearDashboardDetail());
    }
    if (els.detailBody) {
      els.detailBody.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-jobtrack-detail-entry]');
        if (!button || !els.detailBody.contains(button)) return;
        const entryId = button.dataset.jobtrackDetailEntry;
        if (!entryId) return;
        const entry = state.entryItems.get(entryId)
          || state.dashboardEntries.find(item => item.applicationId === entryId);
        if (!entry) return;
        openDetailModal(entry);
      });
    }
    if (els.detailModalClose) {
      els.detailModalClose.addEventListener('click', () => closeDetailModal());
    }
    if (els.detailModal) {
      els.detailModal.addEventListener('click', (event) => {
        if (event.target === els.detailModal) closeDetailModal();
      });
    }
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!els.detailModal || !els.detailModal.classList.contains('active')) return;
      event.preventDefault();
      closeDetailModal();
    });
  };

	  const initAuth = async () => {
	    updateConfigStatus();
	    const storedRaw = loadAuth();
	    const stored = normalizeAuth(storedRaw);
	    if (authIsValid(stored)) {
	      state.auth = stored;
	      saveAuth(stored);
	    } else if (storedRaw) {
	      clearAuth();
	      setAuthMessage('Your session ended. Please sign in again.', 'info');
	    }
	    try {
	      await handleAuthRedirect();
	    } catch (err) {
	      console.error('Auth redirect failed', err);
	      setAuthMessage(err?.message || 'Sign-in failed.', 'error');
	    }
	    updateAuthUI();
	    if (els.signIn) {
	      els.signIn.addEventListener('click', async () => {
	        const hash = window.location && window.location.hash
	          ? window.location.hash.replace('#', '')
	          : '';
	        const selectedTab = tabs.buttons.find(button => button.getAttribute('aria-selected') === 'true')?.dataset.jobtrackTab || '';
	        const returnTab = tabs.buttons.some(button => button.dataset.jobtrackTab === hash) ? hash : selectedTab;
	        try {
	          sessionStorage.setItem(RETURN_TAB_KEY, returnTab || 'account');
	        } catch {}

	        if (!config.cognitoDomain || !config.cognitoClientId || !config.cognitoRedirect) {
	          setAuthMessage('Cognito settings are missing.', 'error');
	          updateAuthUI();
	          return;
	        }
	        try {
	          setAuthMessage('Redirecting to sign-in...', 'info');
	          updateAuthUI();
	          const url = await buildAuthorizeUrl();
	          window.location.assign(url);
	        } catch (err) {
	          console.error('Sign-in failed', err);
	          setAuthMessage(err?.message || 'Unable to start sign-in.', 'error');
	          updateAuthUI();
	        }
	      });
	    }
	    if (els.signOut) {
	      els.signOut.addEventListener('click', () => {
	        clearAuth();
	        setAuthMessage('Signed out. Sign in to continue.', 'info');
	        clearEntryEditMode('Sign in to save entries.', 'info');
	        if (els.entryForm) {
	          state.isResettingEntry = true;
	          els.entryForm.reset();
          state.isResettingEntry = false;
          setEntryType('application');
          resetEntryDateFields('application');
          clearAttachmentInputs();
          clearCustomFields();
	        }
	        updateAuthUI();
	        refreshDashboard();
	        refreshEntries();
	      });
	    }
	    startAuthWatcher();
	  };

	  const init = async () => {
	    initTabs();
	    initJumpButtons();
	    initFilters();
	    initDashboardView();
	    initDashboardInteractions();
	    initEntryForm();
	    initImport();
	    initProspectImport();
    initEntryList();
    initProspectReview();
    initExport();
    await initAuth();
    updateAuthUI();
    refreshDashboard();
    refreshEntries();
  };

  init();
})();
