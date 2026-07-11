'use strict';

(function initActivityEvents() {
  const body = document.body;
  const pathParts = window.location.pathname.split('/').filter(Boolean);
  const directorySearchTimers = new WeakMap();
  const directorySearchValues = new WeakMap();
  const directoryDepthTracked = new WeakSet();
  let pendingToolRun = null;
  let gameSessionStarted = false;
  let caseStudyTimer = null;
  let caseStudyTracked = false;

  function safeId(value, fallback = 'unknown') {
    const normalized = String(value || '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 64);
    return normalized || fallback;
  }

  function lengthBucket(value) {
    const length = String(value || '').trim().length;
    if (length === 0) return '0';
    if (length <= 10) return '1-10';
    if (length <= 25) return '11-25';
    if (length <= 50) return '26-50';
    if (length <= 100) return '51-100';
    return '101-plus';
  }

  function tokenBucket(value) {
    const count = String(value || '').trim().split(/\s+/).filter(Boolean).length;
    if (count === 0) return '0';
    if (count === 1) return '1';
    if (count <= 3) return '2-3';
    if (count <= 6) return '4-6';
    return '7-plus';
  }

  function resultBucket(count) {
    const value = Math.max(0, Number(count) || 0);
    if (value === 0) return '0';
    if (value <= 5) return '1-5';
    if (value <= 10) return '6-10';
    return '11-plus';
  }

  function durationBucket(durationMs) {
    const value = Math.max(0, Number(durationMs) || 0);
    if (value < 1000) return 'under-1s';
    if (value < 3000) return '1-3s';
    if (value < 10000) return '3-10s';
    if (value < 30000) return '10-30s';
    if (value < 120000) return '30-120s';
    return '120s-plus';
  }

  function currentTimeMs() {
    if (window.performance && typeof window.performance.now === 'function') {
      return window.performance.now();
    }
    return Date.now();
  }

  function safeToolErrorType(value) {
    const type = safeId(value, 'unknown');
    return [
      'network', 'permission', 'processing', 'runtime', 'service',
      'timeout', 'unsupported', 'validation', 'unknown'
    ].includes(type) ? type : 'unknown';
  }

  function emit(name, params = {}) {
    if (typeof window.gaEvent !== 'function') return false;
    return window.gaEvent(name, params);
  }

  function clickSurface(element) {
    if (!element) return 'unknown';
    if (element.closest('.portfolio-brand-panel')) return 'portfolio_header';
    if (element.closest('.home-graph')) return 'home_explorer';
    if (element.closest('.contact-card')) return 'contact_card';
    if (element.closest('[data-speed-dial-action], .speed-dial')) return 'speed_dial';
    if (element.closest('.nav-dropdown')) return 'nav_dropdown';
    if (element.closest('header, .site-header')) return 'header';
    if (element.closest('footer, .site-footer')) return 'footer';
    if (element.closest('main')) return 'main';
    return 'other';
  }

  function resumeVariant(link) {
    const href = String(link?.getAttribute('href') || '').toLowerCase();
    if (href.includes('data-science')) return 'data_science';
    if (href.includes('tourism')) return 'tourism';
    if (href.includes('analytics')) return 'analytics';
    return 'general';
  }

  function contactMethod(link) {
    const href = String(link?.getAttribute('href') || '').toLowerCase();
    if (href.startsWith('mailto:')) return 'email';
    if (href.includes('linkedin.com')) return 'linkedin';
    if (href.includes('github.com')) return 'github';
    return 'message_form';
  }

  function resumeActionType(link) {
    const href = String(link?.getAttribute('href') || '').toLowerCase().split(/[?#]/)[0];
    if (link?.hasAttribute?.('download')) return 'download_pdf';
    if (/(?:^|\/)resume(?:-[a-z-]+)?-pdf(?:\.html)?$/.test(href)) return 'preview_pdf';
    if (/(?:^|\/)documents\/resume[^/]*\.pdf$/.test(href)) return 'download_pdf';
    return 'view_html';
  }

  function contentResourceType(link) {
    const explicit = safeId(link?.dataset?.resourceType || '', '');
    if (explicit) return explicit;
    const href = String(link?.getAttribute?.('href') || '').toLowerCase().split(/[?#]/)[0];
    const label = String(link?.textContent || '').toLowerCase();
    const signal = `${href} ${label}`;
    if (href.includes('github.com')) return 'github';
    if (/\.ipynb$|\bnotebook\b/.test(signal)) return 'notebook';
    if (/\.pdf$|\bpdf\b/.test(signal)) return 'pdf';
    if (/\.xlsx?$|\bexcel\b|\bspreadsheet\b/.test(signal)) return 'spreadsheet';
    if (/\.csv$|\bdataset\b|kaggle/.test(signal)) return 'dataset';
    if (/tableau|powerbi|dashboard/.test(signal)) return 'dashboard';
    if (/\blive[- ]?demo\b|\binteractive\b|(?:^|\/)demos?\//.test(signal)) return 'live_demo';
    if (/(?:^|\/)portfolio\//.test(href)) return 'case_study';
    if (/(?:^|\/)tools\//.test(href)) return 'tool';
    if (/(?:^|\/)games\//.test(href)) return 'game';
    return 'resource';
  }

  function directoryRoot(element) {
    return element?.closest?.('[data-portfolio-workbench]') || null;
  }

  function directoryType(root) {
    if (!root) return 'unknown';
    return safeId(root.dataset.directoryWorkbench || 'portfolio');
  }

  function visibleDirectoryResultCount(root) {
    return Array.from(root?.querySelectorAll?.('[data-portfolio-results] [data-project-id]') || [])
      .filter((entry) => !entry.hidden).length;
  }

  function setupDirectoryDepthTracking() {
    document.querySelectorAll('[data-portfolio-workbench]').forEach((root) => {
      const results = root.querySelector('[data-portfolio-results]');
      if (!results) return;
      results.addEventListener('scroll', () => {
        if (directoryDepthTracked.has(results)) return;
        const scrollable = Math.max(0, Number(results.scrollHeight || 0) - Number(results.clientHeight || 0));
        if (scrollable <= 0) return;
        const percent = Math.round((Math.max(0, Number(results.scrollTop || 0)) / scrollable) * 100);
        if (percent < 50) return;
        const sent = emit('directory_depth_reached', {
          directory_type: directoryType(root),
          source_surface: 'directory_results',
          percent: 50,
          result_bucket: resultBucket(visibleDirectoryResultCount(root))
        });
        if (sent) directoryDepthTracked.add(results);
      }, { passive: true });
    });
  }

  function trackDirectoryFilter(target) {
    const root = directoryRoot(target);
    if (!root) return;

    if (target.matches('[data-portfolio-sort]')) {
      const visibleResults = visibleDirectoryResultCount(root);
      emit('directory_filter_apply', {
        directory_type: directoryType(root),
        filter_group: 'sort',
        filter_value: safeId(target.value),
        filter_state: 'selected',
        result_bucket: resultBucket(visibleResults)
      });
      return;
    }

    if (!target.closest('[data-portfolio-filters]')) return;
    const visibleResults = visibleDirectoryResultCount(root);
    emit('directory_filter_apply', {
      directory_type: directoryType(root),
      filter_group: safeId(target.name || target.closest('fieldset')?.querySelector('[data-portfolio-filter-toggle]')?.dataset.portfolioFilterToggle),
      filter_value: safeId(target.value),
      filter_state: target.checked === false ? 'removed' : 'selected',
      result_bucket: resultBucket(visibleResults)
    });
  }

  function trackDirectoryFilterClick(target) {
    const root = directoryRoot(target);
    if (!root) return false;
    let group = '';
    let value = '';
    let state = '';

    if (target.closest('[data-portfolio-clear-filters], [data-portfolio-mobile-clear]')) {
      group = 'all';
      value = 'all';
      state = 'cleared';
    } else {
      const scope = target.closest('[data-portfolio-scope-toggle]');
      const quick = target.closest('[data-portfolio-mobile-quick-filter]');
      const remove = target.closest('[data-portfolio-mobile-remove-filter]');
      if (scope) {
        group = 'audience_scope';
        value = scope.getAttribute('aria-pressed') === 'true' ? 'audience_only' : 'all';
        state = 'selected';
      } else if (quick) {
        group = quick.dataset.mobileFilterGroup;
        value = quick.dataset.mobileFilterValue;
        state = quick.getAttribute('aria-pressed') === 'true' ? 'removed' : 'selected';
      } else if (remove) {
        group = remove.dataset.mobileFilterGroup;
        value = remove.dataset.mobileFilterValue;
        state = 'removed';
      }
    }

    if (!group || !value || !state) return false;
    window.setTimeout(() => {
      emit('directory_filter_apply', {
        directory_type: directoryType(root),
        filter_group: safeId(group),
        filter_value: safeId(value),
        filter_state: state,
        result_bucket: resultBucket(visibleDirectoryResultCount(root))
      });
    }, 0);
    return true;
  }

  function scheduleDirectorySearch(input) {
    const root = directoryRoot(input);
    if (!root) return;
    const query = String(input.value || '').trim();
    const priorTimer = directorySearchTimers.get(input);
    if (priorTimer) window.clearTimeout(priorTimer);
    if (!query) return;

    const timer = window.setTimeout(() => {
      const currentQuery = String(input.value || '').trim();
      if (!currentQuery || currentQuery !== query || directorySearchValues.get(input) === currentQuery) return;
      const visibleResults = visibleDirectoryResultCount(root);
      const sent = emit('directory_search', {
        directory_type: directoryType(root),
        query_length_bucket: lengthBucket(currentQuery),
        query_token_bucket: tokenBucket(currentQuery),
        result_bucket: resultBucket(visibleResults),
        has_results: visibleResults > 0
      });
      if (sent) directorySearchValues.set(input, currentQuery);
    }, 650);
    directorySearchTimers.set(input, timer);
  }

  function trackSelectedContent(target) {
    const card = target.closest('[data-project-id]');
    if (!card) return false;
    const root = directoryRoot(card);
    const rootType = directoryType(root);
    const contentType = rootType === 'tools' ? 'tool' : rootType === 'games' ? 'game' : 'project';
    emit('select_content', {
      content_type: contentType,
      item_id: safeId(card.dataset.projectId),
      source_surface: 'directory_results'
    });
    if (root && rootType === 'portfolio' && typeof window.trackProjectView === 'function') {
      window.trackProjectView(card.dataset.projectId, { source_surface: 'portfolio_workbench' });
    }
    return true;
  }

  function trackContentOpen(target) {
    const explicit = target.closest('[data-content-open][href]');
    const projectResource = target.closest('.project-link[href]');
    const link = explicit || projectResource;
    if (!link) return false;

    let contentId = explicit?.dataset?.contentId || '';
    let contentType = explicit?.dataset?.contentType || '';
    let sourceSurface = explicit?.dataset?.sourceSurface || '';

    if (!explicit) {
      if (pathParts[0] !== 'portfolio' || pathParts.length < 2) return false;
      contentId = pathParts[1].replace(/\.html$/i, '');
      contentType = 'project_resource';
      sourceSurface = 'case_study_resources';
    }

    const normalizedContentType = safeId(contentType, 'content');
    const normalizedContentId = safeId(contentId);
    const normalizedSourceSurface = safeId(sourceSurface, 'content_link');
    emit('select_content', {
      content_type: normalizedContentType,
      content_id: normalizedContentId,
      resource_type: contentResourceType(link),
      source_surface: normalizedSourceSurface
    });
    if (normalizedContentType === 'project' && typeof window.trackProjectView === 'function') {
      window.trackProjectView(normalizedContentId, { source_surface: normalizedSourceSurface });
    }
    return true;
  }

  function trackHomeExplore(target) {
    const graphRoot = target.closest('[data-home-graph]');
    if (!graphRoot) return false;

    const introLink = target.closest('.home-graph__intro-action[href]');
    if (introLink) {
      const href = String(introLink.getAttribute('href') || '').split(/[?#]/)[0];
      emit('home_explore_select', {
        explore_type: 'directory',
        item_id: safeId(href.split('/').filter(Boolean).pop() || 'home')
      });
      return true;
    }

    const item = target.closest('[data-graph-item], [data-graph-dot-item]');
    const group = target.closest('[data-graph-group]');
    const category = target.closest('[data-graph-category-button], [data-graph-category]');
    if (item) {
      emit('home_explore_select', {
        explore_type: 'item',
        item_id: safeId(item.dataset.itemId || item.dataset.graphDotItem || item.dataset.graphItem)
      });
      return true;
    }
    if (group) {
      emit('home_explore_select', {
        explore_type: 'group',
        item_id: safeId(group.dataset.groupId || group.dataset.graphGroup)
      });
      return true;
    }
    if (category) {
      emit('home_explore_select', {
        explore_type: 'category',
        item_id: safeId(category.dataset.graphCategoryButton || category.dataset.graphCategory)
      });
      return true;
    }
    return false;
  }

  function toolContext() {
    const main = document.querySelector('main');
    const bodyPage = safeId(body?.dataset?.page || '');
    const pathIsTool = pathParts[0] === 'tools' && pathParts.length > 1;
    const hasToolPageClass = Boolean(body?.classList?.contains('tools-tool-page'));
    if (!main || (!pathIsTool && !hasToolPageClass) || ['tools', 'tools-dashboard'].includes(bodyPage)) return null;
    return { main, toolId: safeId(pathIsTool ? pathParts[1] : bodyPage) };
  }

  function controlSignal(element) {
    if (!element) return '';
    return [
      element.dataset?.action,
      element.dataset?.toolsAction,
      element.id,
      element.getAttribute?.('name'),
      element.className,
      element.getAttribute?.('aria-label'),
      element.textContent
    ].map((value) => String(value || '').toLowerCase()).join(' ');
  }

  function toolAction(element) {
    const signal = controlSignal(element);
    if (/transcrib/.test(signal)) return 'transcribe';
    if (/optimi[sz]/.test(signal)) return 'optimize';
    if (/compar/.test(signal)) return 'compare';
    if (/clean/.test(signal)) return 'clean';
    if (/generat|create/.test(signal)) return 'generate';
    if (/record/.test(signal)) return 'record';
    if (/remov/.test(signal)) return 'remove';
    if (/analy[sz]|check/.test(signal)) return 'analyze';
    if (/upload|process/.test(signal)) return 'process';
    return 'run';
  }

  function startsToolRun(element) {
    const signal = controlSignal(element);
    if (/download|export|copy|save session|clear|reset|cancel|close|delete|remove file/.test(signal)) return false;
    return /\brun\b|transcrib|optimi[sz]|compar|clean|generat|create qr|start record|remove background|analy[sz]|\bcheck\b|\bprocess\b/.test(signal);
  }

  function beginToolRun(element) {
    const context = toolContext();
    if (!context || !context.main.contains(element)) return;
    const action = toolAction(element);
    startToolRun(context.toolId, action, { restart: true });
  }

  function startToolRun(toolId, action = 'run', options = {}) {
    const context = toolContext();
    const normalizedToolId = safeId(toolId || context?.toolId);
    if (!context || normalizedToolId !== context.toolId) return false;
    if (pendingToolRun?.toolId === normalizedToolId && options.restart !== true) return true;
    const normalizedAction = safeId(action, 'run');
    const sent = emit('tool_run_start', { tool_id: normalizedToolId, action: normalizedAction });
    pendingToolRun = sent ? {
      toolId: normalizedToolId,
      action: normalizedAction,
      startedAt: currentTimeMs()
    } : null;
    return sent;
  }

  function finishToolRun(eventName, detail = {}) {
    if (!pendingToolRun) return false;
    const detailToolId = safeId(detail.toolId || detail.tool_id || pendingToolRun.toolId);
    if (!detailToolId || detailToolId === 'unknown' || detailToolId !== pendingToolRun.toolId) return false;

    const action = safeId(detail.action || pendingToolRun?.action || 'run');
    const params = {
      tool_id: detailToolId,
      action
    };
    if (pendingToolRun && Number.isFinite(pendingToolRun.startedAt)) {
      params.duration_bucket = durationBucket(currentTimeMs() - pendingToolRun.startedAt);
    }
    if (eventName === 'tool_run_error') {
      params.error_type = safeToolErrorType(detail.errorType || detail.error_type);
    } else if (detail.resultBucket || detail.result_bucket) {
      params.result_bucket = safeId(detail.resultBucket || detail.result_bucket);
    }

    const sent = emit(eventName, params);
    if (pendingToolRun && detailToolId === pendingToolRun.toolId) pendingToolRun = null;
    return sent;
  }

  function exportAction(element) {
    const signal = controlSignal(element);
    if (/copy/.test(signal)) return 'copy';
    if (/csv/.test(signal)) return 'export_csv';
    if (/json/.test(signal)) return 'export_json';
    if (/download/.test(signal)) return 'download';
    if (/export/.test(signal)) return 'export';
    return '';
  }

  function trackToolExport(target) {
    const context = toolContext();
    if (!context || !context.main.contains(target)) return false;
    const control = target.closest('a[download], button, [role="button"]');
    if (!control || control.disabled || control.getAttribute('aria-disabled') === 'true') return false;
    const action = exportAction(control);
    if (!action) return false;
    emit('tool_output_export', { tool_id: context.toolId, action });
    return true;
  }

  function gameId() {
    if (pathParts[0] !== 'games' || pathParts.length < 2) return '';
    return safeId(pathParts[1].replace(/\.html$/i, ''));
  }

  function setupCaseStudyEngagement() {
    if (pathParts[0] !== 'portfolio' || pathParts.length < 2 || typeof IntersectionObserver !== 'function') return;
    const proofSection = document.querySelector('.project-star');
    if (!proofSection) return;

    let sufficientlyVisible = false;
    const observer = new IntersectionObserver((entries) => {
      sufficientlyVisible = entries.some((entry) => entry.isIntersecting && entry.intersectionRatio >= 0.5);
      if (!sufficientlyVisible || caseStudyTracked) {
        if (caseStudyTimer !== null) window.clearTimeout(caseStudyTimer);
        caseStudyTimer = null;
        return;
      }

      if (caseStudyTimer !== null) return;
      caseStudyTimer = window.setTimeout(() => {
        caseStudyTimer = null;
        if (!sufficientlyVisible || caseStudyTracked || document.visibilityState === 'hidden') return;
        const projectId = safeId(pathParts[1].replace(/\.html$/i, ''));
        caseStudyTracked = emit('case_study_engaged', {
          project_id: projectId
        });
        if (caseStudyTracked && typeof window.trackProjectView === 'function') {
          window.trackProjectView(projectId, { source_surface: 'case_study' });
        }
        if (caseStudyTracked) observer.disconnect();
      }, 5000);
    }, { threshold: [0.5] });

    observer.observe(proofSection);
  }

  function startGameSession(event) {
    if (gameSessionStarted) return;
    const id = gameId();
    if (!id) return;
    const main = document.querySelector('main');
    const target = event.target;
    if (!main || !target || !main.contains(target)) return;
    if (target.closest('a[href], .site-chatbot, #pcz-banner')) return;

    if (event.type === 'keydown') {
      const key = String(event.key || '');
      if (target.matches('input, textarea, select') || ['Tab', 'Shift', 'Control', 'Alt', 'Meta', 'Escape'].includes(key)) return;
    }

    const inputType = event.type === 'keydown'
      ? 'keyboard'
      : event.pointerType === 'touch' ? 'touch' : 'pointer';
    gameSessionStarted = emit('game_session_start', { game_id: id, input_type: inputType });
  }

  document.addEventListener('submit', (event) => {
    const context = toolContext();
    if (!context || !context.main.contains(event.target)) return;
    const trigger = event.submitter || event.target;
    if (!startsToolRun(trigger)) return;
    beginToolRun(trigger);
  }, true);

  document.addEventListener('tools:run-complete', (event) => {
    finishToolRun('tool_run_complete', event?.detail || {});
  });

  document.addEventListener('tools:run-error', (event) => {
    finishToolRun('tool_run_error', event?.detail || {});
  });

  document.addEventListener('tools:run-start', (event) => {
    const detail = event?.detail || {};
    startToolRun(detail.toolId || detail.tool_id, detail.action || 'run');
  });

  document.addEventListener('tools:run-cancel', (event) => {
    if (!pendingToolRun) return;
    const detail = event?.detail || {};
    const detailToolId = safeId(detail.toolId || detail.tool_id || pendingToolRun.toolId);
    if (detailToolId === pendingToolRun.toolId) pendingToolRun = null;
  });

  document.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.matches('[data-portfolio-sort], [data-portfolio-filters] input, [data-portfolio-filters] select')) {
      trackDirectoryFilter(target);
    }
  });

  document.addEventListener('input', (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || !input.matches('[data-portfolio-search]')) return;
    scheduleDirectorySearch(input);
  });

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const audienceLink = target.closest('[data-portfolio-audience-link]');
    if (audienceLink) {
      emit('portfolio_audience_select', {
        current_audience: safeId(new URLSearchParams(window.location.search).get('audience') || 'personal'),
        selected_audience: safeId(audienceLink.dataset.portfolioAudienceLink, 'personal')
      });
    }

    const resumeLink = target.closest('[data-resume-home-link], [data-portfolio-resume-link], a[href*="resume" i]');
    if (resumeLink) {
      emit('resume_cta_click', {
        resume_variant: resumeVariant(resumeLink),
        cta_surface: clickSurface(resumeLink),
        action_type: resumeActionType(resumeLink)
      });
    }

    const contactLink = target.closest('[data-contact-modal-link], a[href^="mailto:"]');
    const socialContact = target.closest('.contact-card[href*="linkedin.com"], .contact-card[href*="github.com"], [data-speed-dial-action][href*="linkedin.com"], [data-speed-dial-action][href*="github.com"]');
    const contactTarget = contactLink || socialContact;
    if (contactTarget) {
      emit('contact_intent', {
        contact_method: contactMethod(contactTarget),
        cta_surface: clickSurface(contactTarget)
      });
    }

    trackDirectoryFilterClick(target);
    const openedContent = trackContentOpen(target);
    if (!openedContent) trackSelectedContent(target);
    trackHomeExplore(target);

    const context = toolContext();
    if (context && context.main.contains(target)) {
      if (!trackToolExport(target)) {
        const control = target.closest('button, [role="button"]');
        const isModeControl = control?.matches?.('[role="tab"], [data-qrtool-tab]');
        if (control && !isModeControl && !control.closest('form') && startsToolRun(control)) beginToolRun(control);
      }
    }
  }, true);

  document.addEventListener('pointerdown', startGameSession, true);
  document.addEventListener('keydown', startGameSession, true);
  setupDirectoryDepthTracking();
  setupCaseStudyEngagement();
})();
