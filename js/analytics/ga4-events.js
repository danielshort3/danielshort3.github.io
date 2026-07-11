/* ===================================================================
   File: ga4-events.js
   Purpose: Consent-aware event helpers for the GTM-managed GA4 tag
   =================================================================== */
(() => {
  'use strict';

  window.dataLayer = window.dataLayer || [];

  const AUTO_COLLECTED_EVENTS = new Set([
    'contact_page_view',
    'engaged_time',
    'outbound_click',
    'contact_form_submit',
    'resume_download'
  ]);

  const EVENT_PARAM_ALLOWLIST = Object.freeze({
    hero_cta_click: ['cta_label'],
    project_filter_select: ['filter_name', 'filter_value', 'selected'],
    see_more_toggle: ['expanded'],
    email_cta_click: ['link_url'],
    nav_link_click: ['link_url'],
    contact_card_click: ['card_label', 'card_type'],
    scroll_depth: ['percent'],
    project_view: ['project_id'],
    multi_project_view: ['view_count'],
    modal_close: ['project_id'],
    contact_modal_open: ['page_path'],
    contact_modal_close: ['page_path'],
    contact_form_validation_error: ['page_path', 'field_id'],
    contact_form_success: ['page_path'],
    contact_form_error: ['page_path'],
    contrib_doc_click: ['section', 'title', 'kind'],
    contrib_timeline_toggle: ['section', 'year', 'expanded'],
    client_error: ['kind', 'page_path'],
    chatbot_launcher_opened: ['source'],
    chatbot_starter_prompts_hidden: [],
    chatbot_reset: [],
    chatbot_nudge_shown: [],
    chatbot_nudge_opened: ['reason'],
    chatbot_nudge_dismissed: ['reason'],
    home_explore_select: ['selection_type', 'explore_type', 'content_type', 'content_id', 'item_id', 'source_surface'],
    portfolio_audience_select: ['current_audience', 'selected_audience'],
    directory_filter_apply: ['directory_type', 'filter_group', 'filter_value', 'filter_state', 'action_type', 'result_bucket'],
    directory_search: ['directory_type', 'query_length_bucket', 'query_token_bucket', 'token_count_bucket', 'result_bucket', 'has_results'],
    select_content: ['content_type', 'content_id', 'item_id', 'source_surface'],
    case_study_engaged: ['project_id'],
    resume_cta_click: ['resume_variant', 'source_surface', 'cta_surface'],
    tool_run_start: ['tool_id', 'action', 'action_type'],
    tool_run_complete: ['tool_id', 'action', 'action_type', 'result_bucket', 'duration_bucket'],
    tool_output_export: ['tool_id', 'action', 'action_type', 'export_type', 'result_bucket'],
    tool_session_save: ['tool_id', 'action', 'action_type', 'save_source'],
    game_session_start: ['game_id', 'action_type', 'input_type'],
    game_milestone: ['game_id', 'action_type', 'milestone_id', 'outcome', 'duration_bucket', 'score_bucket'],
    contact_intent: ['method', 'contact_method', 'source_surface', 'cta_surface'],
    generate_lead: ['source_surface'],
    chatbot_question_submit: ['source', 'question_length_bucket', 'question_token_bucket', 'transcript_size_bucket', 'is_followup'],
    chatbot_response_success: ['source', 'answer_length_bucket', 'response_length_bucket', 'source_count', 'source_count_bucket', 'followup_count', 'response_mode'],
    chatbot_response_error: ['source', 'error_type'],
    chatbot_response_stopped: ['source', 'had_partial_response'],
    chatbot_rate_limited: ['source', 'limit_type', 'retry_bucket', 'challenge_required'],
    chatbot_link_click: ['source', 'link_type', 'destination_kind', 'destination_section'],
    site_search: ['query_length_bucket', 'query_token_bucket', 'token_count_bucket', 'result_count', 'result_count_bucket', 'result_bucket', 'has_results', 'search_surface'],
    site_search_result_click: ['result_category', 'result_position', 'result_position_bucket']
  });

  const ACTIVITY_CATEGORY_BY_EVENT = Object.freeze({
    hero_cta_click: 'navigation',
    nav_link_click: 'navigation',
    portfolio_audience_select: 'navigation',
    home_explore_select: 'discovery',
    project_filter_select: 'directory',
    directory_filter_apply: 'directory',
    directory_search: 'directory',
    select_content: 'directory',
    project_view: 'portfolio',
    multi_project_view: 'portfolio',
    modal_close: 'portfolio',
    case_study_engaged: 'portfolio',
    resume_cta_click: 'career_intent',
    email_cta_click: 'career_intent',
    tool_run_start: 'tool_activation',
    tool_run_complete: 'tool_activation',
    tool_output_export: 'tool_value',
    tool_session_save: 'tool_value',
    game_session_start: 'game_activation',
    game_milestone: 'game_progress',
    contact_intent: 'contact',
    contact_modal_open: 'contact',
    contact_modal_close: 'contact',
    contact_form_validation_error: 'contact',
    contact_card_click: 'contact',
    contact_form_success: 'lead',
    contact_form_error: 'reliability',
    chatbot_launcher_opened: 'chatbot_activation',
    chatbot_starter_prompts_hidden: 'chatbot_activation',
    chatbot_nudge_shown: 'chatbot_activation',
    chatbot_nudge_opened: 'chatbot_activation',
    chatbot_nudge_dismissed: 'chatbot_activation',
    chatbot_reset: 'chatbot_activation',
    chatbot_question_submit: 'chatbot_activation',
    chatbot_response_success: 'chatbot_outcome',
    chatbot_response_error: 'chatbot_outcome',
    chatbot_response_stopped: 'chatbot_outcome',
    chatbot_rate_limited: 'chatbot_outcome',
    chatbot_link_click: 'chatbot_outcome',
    site_search: 'site_search',
    site_search_result_click: 'site_search',
    see_more_toggle: 'content',
    scroll_depth: 'content',
    contrib_doc_click: 'content',
    contrib_timeline_toggle: 'content',
    client_error: 'reliability'
  });

  const EMAIL_PATTERN = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;
  const ACTIVITY_LABEL_KEYS = [
    'tool_id', 'game_id', 'project_id', 'content_id', 'item_id', 'filter_value',
    'resume_variant', 'contact_method', 'method', 'link_type', 'field_id',
    'cta_label', 'card_type', 'section', 'result_category', 'directory_type'
  ];
  const ACTIVITY_DETAIL_KEYS = [
    'source_surface', 'cta_surface', 'search_surface', 'filter_group', 'content_type',
    'selection_type', 'explore_type', 'action_type', 'action', 'input_type',
    'export_type', 'save_source', 'response_mode', 'limit_type', 'error_type',
    'kind', 'source', 'destination_kind', 'destination_section'
  ];
  const ACTIVITY_VALUE_KEYS = [
    'result_count', 'source_count', 'followup_count', 'view_count', 'percent',
    'result_position', 'result_bucket', 'result_count_bucket', 'result_position_bucket',
    'duration_bucket', 'score_bucket', 'query_length_bucket', 'question_length_bucket',
    'response_length_bucket', 'answer_length_bucket', 'retry_bucket'
  ];
  const ACTIVITY_STATE_KEYS = [
    'selected_audience', 'current_audience', 'filter_state', 'outcome', 'expanded',
    'selected', 'has_results', 'challenge_required', 'is_followup', 'had_partial_response'
  ];

  let analyticsConsentGranted = false;
  let domListenersBound = false;
  const viewedProjectIds = new Set();
  let sent50 = false;

  function normalizeConsentState(value) {
    return value && value.categories ? value.categories : value;
  }

  function isEmbeddedSameOrigin() {
    try {
      if (window.self === window.top) return false;
      return window.top.location.origin === window.location.origin;
    } catch (err) {
      return false;
    }
  }

  function readConsentState() {
    if (!window.consentAPI || typeof window.consentAPI.get !== 'function') return null;
    try {
      return normalizeConsentState(window.consentAPI.get());
    } catch (err) {
      return null;
    }
  }

  function safeText(value, maxLength = 80) {
    const text = String(value == null ? '' : value)
      .replace(/[\u0000-\u001F\u007F]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    if (!text || EMAIL_PATTERN.test(text)) return '';
    return text.slice(0, maxLength);
  }

  function safeToken(value, maxLength = 64) {
    return safeText(value, maxLength * 2)
      .toLowerCase()
      .replace(/[^a-z0-9._:-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, maxLength);
  }

  function safeLabel(value) {
    return safeText(value, 80);
  }

  function safeBucket(value) {
    return safeToken(value, 32);
  }

  function safeBoolean(value) {
    if (value === true || value === false) return value;
    if (value === 'true') return true;
    if (value === 'false') return false;
    return undefined;
  }

  function safeCount(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) return undefined;
    return Math.min(1000000, Math.round(number));
  }

  function safePercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || number > 100) return undefined;
    return Math.round(number);
  }

  function safeYear(value) {
    const year = String(value == null ? '' : value).trim();
    return /^(?:19|20)\d{2}$/.test(year) ? year : '';
  }

  function stripQueryAndHash(value) {
    const raw = String(value == null ? '' : value);
    const queryIndex = raw.indexOf('?');
    const hashIndex = raw.indexOf('#');
    const cutoff = [queryIndex, hashIndex]
      .filter(index => index >= 0)
      .reduce((lowest, index) => Math.min(lowest, index), raw.length);
    return raw.slice(0, cutoff);
  }

  function safePath(value) {
    const source = String(value == null ? '' : value).trim();
    if (/^mailto:/i.test(source)) return 'mailto:';
    if (/^tel:/i.test(source)) return 'tel:';
    if (/^(?:javascript|data|blob):/i.test(source)) return '';
    const raw = safeText(stripQueryAndHash(source), 400);
    if (!raw) return '';

    try {
      if (typeof URL === 'function') {
        const base = (document && document.baseURI) || (window.location && window.location.href) || 'https://www.danielshort.me/';
        const url = new URL(raw, base);
        if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
        return url.pathname || '/';
      }
    } catch (err) {}

    const withoutOrigin = raw.replace(/^https?:\/\/[^/]+/i, '');
    const path = stripQueryAndHash(withoutOrigin);
    if (!path) return '/';
    return (path.startsWith('/') ? path : `/${path}`).slice(0, 200);
  }

  function safeLinkUrl(value) {
    const source = String(value == null ? '' : value).trim();
    if (/^mailto:/i.test(source)) return 'mailto:';
    if (/^tel:/i.test(source)) return 'tel:';
    if (/^(?:javascript|data|blob):/i.test(source)) return '';
    const raw = safeText(stripQueryAndHash(source), 500);
    if (!raw) return '';

    try {
      if (typeof URL === 'function') {
        const base = (document && document.baseURI) || (window.location && window.location.href) || 'https://www.danielshort.me/';
        const url = new URL(raw, base);
        if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';
        url.search = '';
        url.hash = '';
        return `${url.origin}${url.pathname || '/'}`.slice(0, 300);
      }
    } catch (err) {}

    return stripQueryAndHash(raw).slice(0, 300);
  }

  function safeReason(value) {
    const reason = safeToken(value, 24);
    return ['opened', 'dismissed', 'manual', 'timeout', 'unknown'].includes(reason) ? reason : '';
  }

  const PARAM_SANITIZERS = Object.freeze({
    action: safeToken,
    action_type: safeToken,
    answer_length_bucket: safeBucket,
    card_label: safeLabel,
    card_type: safeToken,
    challenge_required: safeBoolean,
    contact_method: safeToken,
    content_id: safeToken,
    content_type: safeToken,
    cta_label: safeLabel,
    cta_surface: safeToken,
    current_audience: safeToken,
    destination_kind: safeToken,
    destination_section: safeToken,
    directory_type: safeToken,
    duration_bucket: safeBucket,
    error_type: safeToken,
    expanded: safeBoolean,
    explore_type: safeToken,
    export_type: safeToken,
    field_id: safeToken,
    filter_group: safeToken,
    filter_name: safeToken,
    filter_state: safeToken,
    filter_value: safeToken,
    followup_count: safeCount,
    game_id: safeToken,
    had_partial_response: safeBoolean,
    has_results: safeBoolean,
    input_type: safeToken,
    is_followup: safeBoolean,
    item_id: safeToken,
    kind: safeToken,
    limit_type: safeToken,
    link_type: safeToken,
    link_url: safeLinkUrl,
    method: safeToken,
    milestone_id: safeToken,
    outcome: safeToken,
    page_path: safePath,
    percent: safePercent,
    project_id: safeToken,
    query_length_bucket: safeBucket,
    query_token_bucket: safeBucket,
    question_length_bucket: safeBucket,
    question_token_bucket: safeBucket,
    reason: safeReason,
    response_length_bucket: safeBucket,
    response_mode: safeToken,
    result_bucket: safeBucket,
    result_category: safeToken,
    result_count: safeCount,
    result_count_bucket: safeBucket,
    result_position: safeCount,
    result_position_bucket: safeBucket,
    resume_variant: safeToken,
    retry_bucket: safeBucket,
    save_source: safeToken,
    score_bucket: safeBucket,
    search_surface: safeToken,
    section: safeLabel,
    selected: safeBoolean,
    selected_audience: safeToken,
    selection_type: safeToken,
    source: safeToken,
    source_count: safeCount,
    source_count_bucket: safeBucket,
    source_surface: safeToken,
    title: safeLabel,
    token_count_bucket: safeBucket,
    tool_id: safeToken,
    transcript_size_bucket: safeBucket,
    view_count: safeCount,
    year: safeYear
  });

  function getCommonContext(params) {
    const bodyData = document && document.body && document.body.dataset ? document.body.dataset : {};
    const requestedPageId = safeToken(params && params.page_id, 64);
    const requestedAudience = safeToken(params && params.audience, 32);
    return {
      page_id: requestedPageId || safeToken(bodyData.page, 64) || 'unknown',
      audience: requestedAudience || safeToken(bodyData.audience, 32) || 'general'
    };
  }

  function sanitizeEventParams(eventName, params) {
    const source = params && typeof params === 'object' && !Array.isArray(params) ? params : {};
    const sanitized = getCommonContext(source);
    const allowedKeys = EVENT_PARAM_ALLOWLIST[eventName] || [];
    allowedKeys.forEach(key => {
      const sanitizer = PARAM_SANITIZERS[key];
      if (!sanitizer || !Object.prototype.hasOwnProperty.call(source, key)) return;
      const value = sanitizer(source[key]);
      if (value === '' || value === undefined || value === null) return;
      sanitized[key] = value;
    });
    if (eventName === 'portfolio_audience_select' && !sanitized.selected_audience && sanitized.audience) {
      sanitized.selected_audience = sanitized.audience;
    }
    return sanitized;
  }

  function firstActivityValue(params, keys) {
    for (const key of keys) {
      if (Object.prototype.hasOwnProperty.call(params, key)) return params[key];
    }
    return undefined;
  }

  function addActivityFields(eventName, eventData) {
    eventData.activity_category = ACTIVITY_CATEGORY_BY_EVENT[eventName] || 'engagement';
    eventData.activity_label = firstActivityValue(eventData, ACTIVITY_LABEL_KEYS) || eventName;
    const detail = firstActivityValue(eventData, ACTIVITY_DETAIL_KEYS);
    const value = firstActivityValue(eventData, ACTIVITY_VALUE_KEYS);
    const state = firstActivityValue(eventData, ACTIVITY_STATE_KEYS);
    if (detail !== undefined) eventData.activity_detail = detail;
    if (value !== undefined) eventData.activity_value = value;
    if (state !== undefined) eventData.activity_state = state;
  }

  function send(name, params = {}) {
    const eventName = String(name || '').trim();
    if (!analyticsConsentGranted || !/^[A-Za-z][A-Za-z0-9_]{0,39}$/.test(eventName)) return false;
    if (AUTO_COLLECTED_EVENTS.has(eventName)) return false;

    const eventData = {
      event: eventName,
      ...sanitizeEventParams(eventName, params)
    };
    addActivityFields(eventName, eventData);
    window.dataLayer.push(eventData);
    return true;
  }

  function updateConsent(value) {
    const state = normalizeConsentState(value);
    analyticsConsentGranted = !isEmbeddedSameOrigin() && !!(state && state.analytics);
  }

  function getClosest(target, selector) {
    return target && typeof target.closest === 'function' ? target.closest(selector) : null;
  }

  function handleDocumentClick(event) {
    const target = event && event.target;
    const link = getClosest(target, 'a[href]');
    const heroCta = getClosest(target, '.hero-cta');
    if (heroCta) {
      send('hero_cta_click', { cta_label: String(heroCta.textContent || '').trim() });
    }

    const filterOption = getClosest(target, '#filter-menu [data-filter]');
    if (filterOption && filterOption.dataset) {
      send('project_filter_select', { filter_name: filterOption.dataset.filter });
    }

    const seeMore = getClosest(target, '#see-more');
    if (seeMore && seeMore.dataset) {
      send('see_more_toggle', { expanded: seeMore.dataset.expanded !== 'true' });
    }

    if (!link) return;

    const href = String(link.getAttribute('href') || '').trim();
    const linkText = String(link.textContent || '').trim();
    if (/^mailto:/i.test(href)) {
      send('email_cta_click', { link_url: href });
    }

    if (link.classList && link.classList.contains('nav-link') && link.getAttribute('target') !== '_blank') {
      send('nav_link_click', { link_url: href });
    }

    const contactCard = getClosest(link, '.contact-card');
    if (document.body && document.body.dataset.page === 'contact' && contactCard) {
      const cardType = /^mailto:/i.test(href) ? 'email' : (/^tel:/i.test(href) ? 'phone' : 'profile');
      send('contact_card_click', { card_label: linkText, card_type: cardType });
    }
  }

  function handleScroll() {
    if (sent50 || !analyticsConsentGranted) return;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    if (scrollable <= 0) return;
    const pct = (window.scrollY || window.pageYOffset || 0) / scrollable;
    if (pct >= 0.5) sent50 = send('scroll_depth', { percent: 50 });
  }

  function bindDomListeners() {
    if (domListenersBound) return;
    domListenersBound = true;
    document.addEventListener('click', handleDocumentClick);
    window.addEventListener('scroll', handleScroll, { passive: true });
  }

  window.gaEvent = send;

  window.trackProjectView = id => {
    const projectId = String(id || '').trim();
    if (!projectId || !send('project_view', { project_id: projectId })) return false;
    const previousCount = viewedProjectIds.size;
    viewedProjectIds.add(projectId.toLowerCase());
    if (previousCount < 3 && viewedProjectIds.size === 3) {
      send('multi_project_view', { view_count: 3 });
    }
    return true;
  };

  window.trackModalClose = id => send('modal_close', { project_id: id });

  window.addEventListener('consent-changed', event => updateConsent(event.detail));

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindDomListeners, { once: true });
  } else {
    bindDomListeners();
  }

  updateConsent(readConsentState());
})();
