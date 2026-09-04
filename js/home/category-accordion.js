(() => {
  'use strict';

  const root = document.querySelector('[data-home-accordion]');
  if (!root) return;

  const items = [...root.querySelectorAll('[data-home-accordion-item]')];
  const triggers = items
    .map((item) => item.querySelector('[data-home-accordion-trigger]'))
    .filter(Boolean);
  const panelById = new Map(items.map((item) => {
    const id = String(item.dataset.homeAccordionItem || '');
    return [id, item.querySelector('[data-home-accordion-panel]')];
  }));
  const scrollerById = new Map([...panelById].map(([id, panel]) => [
    id,
    panel?.querySelector('[data-home-accordion-scroller]') || null
  ]));
  const timelineScrollerById = new Map([...panelById].map(([id, panel]) => [
    id,
    panel?.querySelector('[data-home-timeline-scroller]') || null
  ]));
  const triggerById = new Map(triggers.map((trigger) => [
    String(trigger.dataset.homeAccordionTrigger || ''),
    trigger
  ]));
  const itemById = new Map(items.map((item) => [
    String(item.dataset.homeAccordionItem || ''),
    item
  ]));
  const libraryViewById = new Map([...root.querySelectorAll('[data-home-library-view]')].map((view) => [
    String(view.dataset.homeLibraryView || ''),
    view
  ]));
  const libraryOpenButtons = [...root.querySelectorAll('[data-home-library-open]')];
  const libraryCloseButtons = [...root.querySelectorAll('[data-home-library-close]')];
  const ids = [...itemById.keys()].filter(Boolean);
  const libraryIds = new Set([...libraryViewById.keys()].filter(Boolean));
  const railLayoutQuery = window.matchMedia('(min-width: 960px) and (min-height: 620px)');
  const compactTransitionQuery = window.matchMedia('(max-width: 959px), (max-height: 619px)');
  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scrollPositions = new Map();
  const documentScrollPositions = new Map();
  const closeTimers = new Map();
  const LIBRARY_ROUTES = Object.freeze({
    projects: '/portfolio',
    tools: '/tools',
    games: '/games'
  });
  const LIBRARY_DOCUMENT_TITLES = Object.freeze({
    projects: 'Data & Machine Learning Portfolio | Daniel Short',
    tools: 'Browser Tools for Writing, Images & Campaigns | Daniel Short',
    games: 'Browser Games & Interactive Simulations | Daniel Short'
  });
  const canonicalLink = document.querySelector('link[rel="canonical"]');
  const overviewDocumentMetadata = Object.freeze({
    title: document.title,
    canonicalHref: canonicalLink?.getAttribute('href') || ''
  });
  const PANEL_TRANSITION_MS = 440;
  const VIEW_TRANSITION_MS = 440;
  const VIEW_EXIT_MS = 176;
  const VIEW_ENTRY_MS = 244;
  const COMPACT_VIEW_EXIT_MS = 140;
  const COMPACT_VIEW_ENTRY_MS = 200;
  const defaultPanel = ids.includes(root.dataset.defaultPanel)
    ? root.dataset.defaultPanel
    : ids[0];
  let activeId = ids.find((id) => triggerById.get(id)?.getAttribute('aria-expanded') === 'true')
    || defaultPanel;
  let isLibraryMode = false;
  let scrollerTabStopFrame = 0;
  let viewTransitionTimer = 0;
  let viewTransitionLocked = false;
  let pendingViewApply = null;
  let queuedLocationHref = '';
  let queuedLocationState = null;
  let lastHandledLocationHref = '';

  function panelIdFromHash() {
    const rawId = String(window.location.hash || '').replace(/^#/, '');
    if (!rawId) return '';
    try {
      const id = decodeURIComponent(rawId);
      return ids.includes(id) ? id : '';
    } catch (error) {
      return '';
    }
  }

  function libraryCategoryFromLocation() {
    try {
      const url = new URL(window.location.href);
      const pathname = url.pathname.replace(/\/+$/, '') || '/';
      const canonicalCategory = Object.keys(LIBRARY_ROUTES)
        .find((id) => LIBRARY_ROUTES[id] === pathname);
      if (canonicalCategory) return canonicalCategory;
      if (url.searchParams.get('view') === 'library') {
        const hashCategory = panelIdFromHash();
        return libraryIds.has(hashCategory) ? hashCategory : 'projects';
      }
    } catch (error) {
      return '';
    }
    return '';
  }

  function scrollPositionKey(id, libraryMode = isLibraryMode) {
    return `${libraryMode ? 'library' : 'overview'}:${id}`;
  }

  function panelScrollTarget(id) {
    if (railLayoutQuery.matches && id === 'about') {
      return timelineScrollerById.get(id) || scrollerById.get(id);
    }
    return scrollerById.get(id);
  }

  function panelScrollRegions(id) {
    return [scrollerById.get(id), timelineScrollerById.get(id)].filter(Boolean);
  }

  function saveScrollPosition(id, libraryMode = isLibraryMode) {
    if (!railLayoutQuery.matches || !id) return;
    const scroller = panelScrollTarget(id);
    if (scroller) scrollPositions.set(scrollPositionKey(id, libraryMode), scroller.scrollTop);
  }

  function restoreScrollPosition(id, libraryMode = isLibraryMode, immediate = false) {
    if (!railLayoutQuery.matches || !id) return;
    const restore = () => {
      const scroller = panelScrollTarget(id);
      if (scroller) scroller.scrollTop = scrollPositions.get(scrollPositionKey(id, libraryMode)) || 0;
      scheduleScrollerTabStopUpdate();
    };
    if (immediate) restore();
    else window.requestAnimationFrame(restore);
  }

  function saveDocumentScrollPosition(id, libraryMode = isLibraryMode) {
    if (railLayoutQuery.matches || !id) return;
    documentScrollPositions.set(scrollPositionKey(id, libraryMode), window.scrollY);
  }

  function restoreDocumentScrollPosition(id, libraryMode = isLibraryMode, fallback = null, immediate = false) {
    if (railLayoutQuery.matches || !id) return;
    const key = scrollPositionKey(id, libraryMode);
    const hasSavedPosition = documentScrollPositions.has(key);
    if (!hasSavedPosition && fallback === null) return;
    const top = hasSavedPosition ? documentScrollPositions.get(key) : fallback;
    const restore = () => {
      window.scrollTo({ top: Math.max(0, Number(top) || 0), behavior: 'auto' });
    };
    if (immediate) restore();
    else window.requestAnimationFrame(restore);
  }

  function canonicalPanelHash(id) {
    return `#${encodeURIComponent(id)}`;
  }

  function updateLocation(id, mode = 'push', libraryMode = isLibraryMode) {
    const url = new URL(window.location.href);
    if (libraryMode) {
      url.pathname = LIBRARY_ROUTES[id] || '/';
      url.searchParams.delete('view');
      url.hash = '';
    } else {
      url.pathname = '/';
      url.searchParams.delete('view');
      url.hash = id;
    }
    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    const view = libraryMode ? 'library' : 'overview';
    const currentState = window.history?.state && typeof window.history.state === 'object'
      ? window.history.state
      : {};
    const nextState = {
      ...currentState,
      homePanel: id,
      homeView: view,
      personalCategory: id,
      personalView: view
    };
    if (nextLocation === currentLocation) {
      const stateMatches = currentState.homePanel === id &&
        currentState.homeView === view &&
        currentState.personalCategory === id &&
        currentState.personalView === view;
      if (!stateMatches && typeof window.history?.replaceState === 'function') {
        window.history.replaceState(nextState, '', url);
      }
      lastHandledLocationHref = window.location.href;
      return;
    }
    const historyMethod = mode === 'replace' ? 'replaceState' : 'pushState';
    if (typeof window.history?.[historyMethod] === 'function') {
      window.history[historyMethod](nextState, '', url);
      lastHandledLocationHref = window.location.href;
    } else {
      window.location.assign(url.toString());
    }
  }

  function normalizePanelHash(id) {
    if (!id || !window.location.hash || window.location.hash === canonicalPanelHash(id)) return;
    updateLocation(id, 'replace');
  }

  function syncDocumentMetadata(id, libraryMode = isLibraryMode) {
    document.title = libraryMode && LIBRARY_DOCUMENT_TITLES[id]
      ? LIBRARY_DOCUMENT_TITLES[id]
      : overviewDocumentMetadata.title;
    if (!canonicalLink || !overviewDocumentMetadata.canonicalHref) return;
    canonicalLink.setAttribute('href', libraryMode && LIBRARY_ROUTES[id]
      ? new URL(LIBRARY_ROUTES[id], overviewDocumentMetadata.canonicalHref).href
      : overviewDocumentMetadata.canonicalHref);
  }

  function getLibraryData(id) {
    const data = window.HOME_LIBRARY_DATA && window.HOME_LIBRARY_DATA[id];
    return data && Array.isArray(data.items) ? data : null;
  }

  function normalizeLibraryHref(value) {
    const href = String(value || '').trim();
    if (!href) return '#';
    if (/^(?:[a-z][a-z0-9+.-]*:|#)/i.test(href)) return href;
    return `/${href.replace(/^\/+/, '')}`;
  }

  function createLibraryArrow() {
    const wrapper = document.createElement('span');
    wrapper.className = 'home-library__arrow';
    wrapper.setAttribute('aria-hidden', 'true');
    wrapper.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"></path></svg>';
    return wrapper;
  }

  function createLibraryMedia(item, categoryId) {
    const media = document.createElement('span');
    media.className = 'home-library__media';
    if (item.image) {
      const image = document.createElement('img');
      image.src = normalizeLibraryHref(item.image);
      image.alt = String(item.imageAlt || '');
      image.width = 640;
      image.height = 360;
      image.loading = 'lazy';
      image.decoding = 'async';
      media.classList.add('home-library__media--preview');
      media.append(image);
    } else if (item.iconHtml) {
      media.setAttribute('aria-hidden', 'true');
      media.classList.add('home-library__media--glyph');
      media.innerHTML = String(item.iconHtml);
    } else {
      media.setAttribute('aria-hidden', 'true');
      media.classList.add('home-library__media--glyph');
      const categoryIcon = triggerById.get(categoryId)?.querySelector('svg');
      if (categoryIcon) media.append(categoryIcon.cloneNode(true));
    }
    return media;
  }

  function createLibraryCard(item, categoryId) {
    const listItem = document.createElement('li');
    listItem.className = 'home-library__item';
    const link = document.createElement('a');
    link.className = 'home-library__card';
    link.href = normalizeLibraryHref(item.href);
    link.dataset.personalTransition = 'detail';
    if (item.external) {
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    }
    if (item.contentType) link.dataset.contentType = String(item.contentType);
    if (item.contentId || item.id) link.dataset.contentId = String(item.contentId || item.id);
    if (item.resourceType || item.contentType) {
      link.dataset.resourceType = String(item.resourceType || item.contentType);
    }
    if (item.contentType && (item.contentId || item.id)) {
      link.dataset.contentOpen = 'true';
      link.dataset.sourceSurface = 'home_library';
    }

    const copy = document.createElement('span');
    copy.className = 'home-library__copy';
    const title = document.createElement('strong');
    title.textContent = String(item.title || 'Explore');
    copy.append(title);
    if (item.summary) {
      const summary = document.createElement('span');
      summary.textContent = String(item.summary);
      copy.append(summary);
    }

    link.append(createLibraryMedia(item, categoryId), copy, createLibraryArrow());
    listItem.append(link);
    return listItem;
  }

  function renderLibrary(id) {
    const view = libraryViewById.get(id);
    if (!view || view.dataset.homeLibraryRendered === 'true') return;
    const list = view.querySelector('[data-home-library-list]');
    if (!list) return;
    const data = getLibraryData(id);
    const itemsForLibrary = data?.items || [];
    const fragment = document.createDocumentFragment();
    itemsForLibrary.forEach((item) => fragment.append(createLibraryCard(item, id)));
    list.replaceChildren(fragment);
    const count = view.querySelector('[data-home-library-count]');
    if (count) count.textContent = String(itemsForLibrary.length);
    view.dataset.homeLibraryRendered = 'true';
    scheduleScrollerTabStopUpdate();
  }

  function updateLibraryViewVisibility() {
    libraryViewById.forEach((view, id) => {
      const visible = isLibraryMode && id === activeId;
      if (visible) {
        renderLibrary(id);
        view.hidden = false;
        view.removeAttribute('inert');
        view.removeAttribute('aria-hidden');
      } else {
        view.hidden = true;
        view.setAttribute('inert', '');
        view.setAttribute('aria-hidden', 'true');
      }
    });
    libraryOpenButtons.forEach((button) => {
      const id = String(button.dataset.homeLibraryOpen || '');
      button.setAttribute('aria-expanded', String(isLibraryMode && activeId === id));
    });
  }

  function updateScrollerTabStops() {
    scrollerTabStopFrame = 0;
    scrollerById.forEach((scroller, id) => {
      const timelineScroller = timelineScrollerById.get(id);
      panelScrollRegions(id).forEach((region) => region.removeAttribute('tabindex'));
      if (id !== activeId || panelById.get(id)?.hidden) return;

      const scrollTarget = railLayoutQuery.matches
        ? panelScrollTarget(id)
        : timelineScroller;
      if (!scrollTarget) return;
      const independentlyScrollable = railLayoutQuery.matches
        ? scrollTarget.scrollHeight > scrollTarget.clientHeight + 1
        : scrollTarget.scrollWidth > scrollTarget.clientWidth + 1;
      if (independentlyScrollable) {
        scrollTarget.setAttribute('tabindex', '0');
      }
    });
  }

  function scheduleScrollerTabStopUpdate() {
    if (scrollerTabStopFrame) window.cancelAnimationFrame(scrollerTabStopFrame);
    scrollerTabStopFrame = window.requestAnimationFrame(updateScrollerTabStops);
  }

  function finishClosingPanel(id) {
    const item = itemById.get(id);
    const panel = panelById.get(id);
    if (!item || !panel || item.classList.contains('is-active')) return;
    panel.hidden = true;
    item.classList.remove('is-closing');
    closeTimers.delete(id);
  }

  function clearCloseTimer(id) {
    const timer = closeTimers.get(id);
    if (timer) window.clearTimeout(timer);
    closeTimers.delete(id);
  }

  function updateTriggerState(trigger, selected) {
    if (!trigger) return;
    const triggerId = String(trigger.dataset.homeAccordionTrigger || '');
    trigger.setAttribute('aria-expanded', String(selected));
    if (selected && triggerId === defaultPanel && !isLibraryMode) {
      trigger.setAttribute('aria-disabled', 'true');
    } else {
      trigger.removeAttribute('aria-disabled');
    }
    trigger.removeAttribute('aria-current');
  }

  function applyPanelState(id, options = {}) {
    const previousId = options.previousId || '';
    const animateOutgoing = options.animateOutgoing === true;
    const animateIncoming = options.animateIncoming === true && !reducedMotionQuery.matches;
    items.forEach((item) => {
      const itemId = String(item.dataset.homeAccordionItem || '');
      const selected = itemId === id;
      const trigger = triggerById.get(itemId);
      const panel = panelById.get(itemId);
      clearCloseTimer(itemId);
      item.classList.toggle('is-active', selected);
      item.classList.toggle('is-activating', selected && animateIncoming);
      updateTriggerState(trigger, selected);
      if (!panel) return;
      if (selected) {
        item.classList.remove('is-closing');
        panel.hidden = false;
        panel.removeAttribute('inert');
        panel.removeAttribute('aria-hidden');
        return;
      }

      panel.setAttribute('inert', '');
      panel.setAttribute('aria-hidden', 'true');
      const animateClose = animateOutgoing &&
        itemId === previousId &&
        !panel.hidden &&
        railLayoutQuery.matches &&
        !isLibraryMode &&
        !reducedMotionQuery.matches;
      if (animateClose) {
        item.classList.add('is-closing');
        closeTimers.set(itemId, window.setTimeout(() => {
          finishClosingPanel(itemId);
        }, PANEL_TRANSITION_MS));
      } else {
        panel.hidden = true;
        item.classList.remove('is-closing');
      }
    });
    updateLibraryViewVisibility();
    scheduleScrollerTabStopUpdate();
  }

  function revealPanelTrigger(id) {
    if (railLayoutQuery.matches || isLibraryMode || !id) return;
    window.requestAnimationFrame(() => {
      triggerById.get(id)?.scrollIntoView({
        behavior: reducedMotionQuery.matches ? 'auto' : 'smooth',
        block: 'start'
      });
    });
  }

  function clearViewTransitionState(options = {}) {
    if (viewTransitionTimer) window.clearTimeout(viewTransitionTimer);
    viewTransitionTimer = 0;
    if (options.completePending === true && typeof pendingViewApply === 'function') {
      pendingViewApply();
    }
    pendingViewApply = null;
    viewTransitionLocked = false;
    root.classList.remove('is-view-changing', 'is-view-leaving', 'is-view-entering');
    [document.documentElement, document.body].filter(Boolean).forEach((node) => {
      delete node.dataset.siteTransitionMode;
      delete node.dataset.siteTransitionCategory;
      delete node.dataset.siteTransitionDirection;
    });
    if (queuedLocationHref) {
      const targetHref = queuedLocationHref;
      const targetState = queuedLocationState;
      queuedLocationHref = '';
      queuedLocationState = null;
      window.requestAnimationFrame(() => {
        if (window.location.href !== targetHref && typeof window.history?.replaceState === 'function') {
          window.history.replaceState(targetState, '', targetHref);
        }
        lastHandledLocationHref = '';
        handleLocationChange();
      });
    }
  }

  function runViewTransition(apply, animate = true, direction = 'replace') {
    if (!animate || reducedMotionQuery.matches) {
      apply();
      return true;
    }
    if (viewTransitionLocked) return false;

    viewTransitionLocked = true;
    [document.documentElement, document.body].filter(Boolean).forEach((node) => {
      node.dataset.siteTransitionMode = 'personal';
      node.dataset.siteTransitionCategory = activeId;
      node.dataset.siteTransitionDirection = direction;
    });
    root.classList.add('is-view-changing');
    let applied = false;
    const applyOnce = () => {
      if (applied) return;
      applied = true;
      if (pendingViewApply === applyOnce) pendingViewApply = null;
      apply();
    };
    pendingViewApply = applyOnce;
    if (typeof document.startViewTransition === 'function') {
      try {
        const transition = document.startViewTransition(applyOnce);
        if (transition?.finished && typeof transition.finished.finally === 'function') {
          transition.finished.catch(() => {}).finally(clearViewTransitionState);
        } else {
          viewTransitionTimer = window.setTimeout(clearViewTransitionState, VIEW_TRANSITION_MS);
        }
        return true;
      } catch {
        applyOnce();
        clearViewTransitionState();
        return true;
      }
    }

    const exitDuration = compactTransitionQuery.matches ? COMPACT_VIEW_EXIT_MS : VIEW_EXIT_MS;
    const entryDuration = compactTransitionQuery.matches ? COMPACT_VIEW_ENTRY_MS : VIEW_ENTRY_MS;
    root.classList.add('is-view-leaving');
    viewTransitionTimer = window.setTimeout(() => {
      viewTransitionTimer = 0;
      root.classList.remove('is-view-leaving');
      applyOnce();
      root.classList.add('is-view-entering');
      viewTransitionTimer = window.setTimeout(clearViewTransitionState, entryDuration);
    }, exitDuration);

    return true;
  }

  function applyLibraryMode(nextLibraryMode, options = {}) {
    const next = Boolean(nextLibraryMode);
    if (next === isLibraryMode && options.force !== true) {
      updateLibraryViewVisibility();
      syncDocumentMetadata(activeId, next);
      return false;
    }

    const apply = () => {
      isLibraryMode = next;
      root.classList.toggle('is-library-mode', next);
      root.dataset.homeView = next ? 'library' : 'overview';
      updateLibraryViewVisibility();
      syncDocumentMetadata(activeId, next);
      updateTriggerState(triggerById.get(activeId), true);
      scheduleScrollerTabStopUpdate();
      if (typeof options.afterApply === 'function') options.afterApply();
    };

    return runViewTransition(apply, options.animate !== false, next ? 'forward' : 'back');
  }

  function selectPanel(id, options = {}) {
    if (!ids.includes(id)) return false;
    if (id === activeId) {
      updateLibraryViewVisibility();
      return false;
    }

    const previousId = activeId;
    saveScrollPosition(previousId);
    saveDocumentScrollPosition(previousId);
    activeId = id;
    root.dataset.activePanel = id;
    applyPanelState(id, {
      previousId,
      animateOutgoing: true,
      animateIncoming: true
    });

    if (options.updateHistory !== false) {
      updateLocation(id, options.historyMode || 'push');
    }
    restoreScrollPosition(id);
    restoreDocumentScrollPosition(id, isLibraryMode, isLibraryMode ? 0 : null);
    if (options.reveal !== false) revealPanelTrigger(id);

    root.dispatchEvent(new CustomEvent('home:category-change', {
      bubbles: true,
      detail: { category: id, view: isLibraryMode ? 'library' : 'overview' }
    }));
    return true;
  }

  function resolveTriggerTarget(currentId, requestedId, fallbackId) {
    if (requestedId === currentId && requestedId !== fallbackId) return fallbackId;
    return requestedId;
  }

  function focusLibraryHeading(id) {
    window.requestAnimationFrame(() => {
      libraryViewById.get(id)?.querySelector('[data-home-library-heading]')?.focus({ preventScroll: true });
    });
  }

  function focusOverviewReturn(id) {
    window.requestAnimationFrame(() => {
      const returnTarget = root.querySelector(`[data-home-library-open="${id}"]`)
        || triggerById.get(id);
      returnTarget?.focus({ preventScroll: true });
    });
  }

  function openLibrary(id) {
    if (!libraryIds.has(id)) return false;
    if (id !== activeId) selectPanel(id, { updateHistory: false, reveal: false });
    saveScrollPosition(id, false);
    saveDocumentScrollPosition(id, false);
    renderLibrary(id);
    const changed = applyLibraryMode(true, {
      afterApply: () => {
        restoreScrollPosition(id, true, true);
        restoreDocumentScrollPosition(id, true, 0, true);
        focusLibraryHeading(id);
        if (!queuedLocationHref) {
          updateLocation(id, 'push', true);
          root.dispatchEvent(new CustomEvent('home:library-change', {
            bubbles: true,
            detail: { category: id, expanded: true }
          }));
        }
      }
    });
    return changed;
  }

  function closeLibrary(options = {}) {
    if (!isLibraryMode) return false;
    const closingId = activeId;
    saveScrollPosition(closingId, true);
    saveDocumentScrollPosition(closingId, true);
    const changed = applyLibraryMode(false, {
      afterApply: () => {
        restoreScrollPosition(closingId, false, true);
        restoreDocumentScrollPosition(closingId, false, 0, true);
        if (options.restoreFocus !== false) focusOverviewReturn(closingId);
        if (!queuedLocationHref) {
          updateLocation(closingId, options.historyMode || 'push', false);
          root.dispatchEvent(new CustomEvent('home:library-change', {
            bubbles: true,
            detail: { category: closingId, expanded: false }
          }));
          root.dispatchEvent(new CustomEvent('home:category-change', {
            bubbles: true,
            detail: { category: closingId, view: 'overview' }
          }));
        }
      }
    });
    return changed;
  }

  function activatePanelTrigger(id) {
    if (!ids.includes(id)) return false;
    if (isLibraryMode && id === activeId) {
      return closeLibrary({ restoreFocus: false });
    }
    const isCondensing = id === activeId && id !== defaultPanel;
    const nextId = resolveTriggerTarget(activeId, id, defaultPanel);
    const changed = selectPanel(nextId);
    if (changed && isCondensing) {
      window.requestAnimationFrame(() => {
        triggerById.get(defaultPanel)?.focus({ preventScroll: true });
      });
    }
    return changed;
  }

  function focusRelativeTrigger(current, direction) {
    if (isLibraryMode) {
      triggerById.get(activeId)?.focus();
      return;
    }
    const index = triggers.indexOf(current);
    if (index < 0) return;
    const nextIndex = (index + direction + triggers.length) % triggers.length;
    triggers[nextIndex]?.focus();
  }

  triggers.forEach((trigger) => {
    trigger.addEventListener('click', () => {
      activatePanelTrigger(String(trigger.dataset.homeAccordionTrigger || ''));
    });

    trigger.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
        event.preventDefault();
        focusRelativeTrigger(trigger, 1);
      } else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
        event.preventDefault();
        focusRelativeTrigger(trigger, -1);
      } else if (event.key === 'Home') {
        event.preventDefault();
        (isLibraryMode ? triggerById.get(activeId) : triggers[0])?.focus();
      } else if (event.key === 'End') {
        event.preventDefault();
        (isLibraryMode ? triggerById.get(activeId) : triggers[triggers.length - 1])?.focus();
      } else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activatePanelTrigger(String(trigger.dataset.homeAccordionTrigger || ''));
      }
    });
  });

  libraryOpenButtons.forEach((control) => {
    control.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof event.button === 'number' && event.button !== 0) return;
      event.preventDefault();
      openLibrary(String(control.dataset.homeLibraryOpen || ''));
    });
  });

  libraryCloseButtons.forEach((button) => {
    button.addEventListener('click', () => closeLibrary());
  });

  function handleLocationChange() {
    const currentHref = window.location.href;
    if (viewTransitionLocked) {
      queuedLocationHref = currentHref;
      queuedLocationState = window.history?.state || null;
      return;
    }
    if (currentHref === lastHandledLocationHref) return;
    lastHandledLocationHref = currentHref;
    const nextLibraryCategory = libraryCategoryFromLocation();
    const nextLibraryMode = Boolean(nextLibraryCategory);
    const hashPanel = panelIdFromHash();
    if (!nextLibraryMode && window.location.hash && !hashPanel) return;
    const nextPanel = nextLibraryCategory || hashPanel || defaultPanel;
    const modeChanged = nextLibraryMode !== isLibraryMode;
    const panelChanged = nextPanel !== activeId;
    if (modeChanged || panelChanged) {
      saveScrollPosition(activeId, isLibraryMode);
      saveDocumentScrollPosition(activeId, isLibraryMode);
    }
    if (nextPanel !== activeId) {
      selectPanel(nextPanel, { updateHistory: false, reveal: !nextLibraryMode });
    }
    const restoreLocationState = () => {
      restoreScrollPosition(nextPanel, nextLibraryMode, true);
      restoreDocumentScrollPosition(nextPanel, nextLibraryMode, nextLibraryMode ? 0 : null, true);
      if (modeChanged) {
        if (nextLibraryMode) focusLibraryHeading(nextPanel);
        else focusOverviewReturn(nextPanel);
      }
      if (nextLibraryMode) updateLocation(nextPanel, 'replace', true);
      if (!nextLibraryMode && hashPanel) normalizePanelHash(hashPanel);
      else if (!nextLibraryMode && !window.location.hash) updateLocation(nextPanel, 'replace', false);
    };
    applyLibraryMode(nextLibraryMode, {
      animate: modeChanged,
      afterApply: modeChanged ? restoreLocationState : null
    });
    if (!modeChanged) restoreLocationState();
  }

  window.addEventListener('hashchange', handleLocationChange);
  window.addEventListener('popstate', handleLocationChange);

  const settleClosingPanels = () => {
    ids.forEach((id) => {
      clearCloseTimer(id);
      finishClosingPanel(id);
    });
    scheduleScrollerTabStopUpdate();
  };

  function listenForMediaChange(query, listener) {
    if (typeof query.addEventListener === 'function') {
      query.addEventListener('change', listener);
    } else if (typeof query.addListener === 'function') {
      query.addListener(listener);
    }
  }

  listenForMediaChange(railLayoutQuery, settleClosingPanels);
  listenForMediaChange(reducedMotionQuery, () => {
    settleClosingPanels();
    if (reducedMotionQuery.matches) clearViewTransitionState({ completePending: true });
  });
  window.addEventListener('resize', scheduleScrollerTabStopUpdate);
  root.addEventListener('load', scheduleScrollerTabStopUpdate, true);
  if (typeof window.ResizeObserver === 'function') {
    const scrollerResizeObserver = new window.ResizeObserver(scheduleScrollerTabStopUpdate);
    ids.forEach((id) => {
      panelScrollRegions(id).forEach((region) => scrollerResizeObserver.observe(region));
    });
  }

  const initialLibraryCategory = libraryCategoryFromLocation();
  const initialLibraryMode = Boolean(initialLibraryCategory);
  const initialHashPanel = panelIdFromHash();
  const initialPanel = initialLibraryCategory || initialHashPanel || defaultPanel;
  activeId = initialPanel;
  root.dataset.activePanel = initialPanel;
  ids.forEach((id) => panelScrollRegions(id).forEach((region) => region.removeAttribute('tabindex')));
  applyPanelState(initialPanel);
  applyLibraryMode(initialLibraryMode, { animate: false, force: true });
  if (initialLibraryMode) {
    updateLocation(initialPanel, 'replace', true);
  } else if (initialHashPanel) {
    normalizePanelHash(initialHashPanel);
    revealPanelTrigger(initialHashPanel);
  } else if (!window.location.hash) {
    updateLocation(initialPanel, 'replace', false);
  }
  root.dataset.enhanced = 'true';
})();
