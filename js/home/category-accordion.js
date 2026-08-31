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
  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scrollPositions = new Map();
  const documentScrollPositions = new Map();
  const closeTimers = new Map();
  const PANEL_TRANSITION_MS = 520;
  const VIEW_TRANSITION_MS = 440;
  const defaultPanel = ids.includes(root.dataset.defaultPanel)
    ? root.dataset.defaultPanel
    : ids[0];
  let activeId = ids.find((id) => triggerById.get(id)?.getAttribute('aria-expanded') === 'true')
    || defaultPanel;
  let isLibraryMode = false;
  let scrollerTabStopFrame = 0;
  let viewTransitionTimer = 0;

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

  function locationRequestsLibrary() {
    try {
      return new URL(window.location.href).searchParams.get('view') === 'library';
    } catch (error) {
      return false;
    }
  }

  function scrollPositionKey(id, libraryMode = isLibraryMode) {
    return `${libraryMode ? 'library' : 'overview'}:${id}`;
  }

  function saveScrollPosition(id, libraryMode = isLibraryMode) {
    if (!railLayoutQuery.matches || !id) return;
    const scroller = scrollerById.get(id);
    if (scroller) scrollPositions.set(scrollPositionKey(id, libraryMode), scroller.scrollTop);
  }

  function restoreScrollPosition(id, libraryMode = isLibraryMode) {
    if (!railLayoutQuery.matches || !id) return;
    window.requestAnimationFrame(() => {
      const scroller = scrollerById.get(id);
      if (scroller) scroller.scrollTop = scrollPositions.get(scrollPositionKey(id, libraryMode)) || 0;
      scheduleScrollerTabStopUpdate();
    });
  }

  function saveDocumentScrollPosition(id, libraryMode = isLibraryMode) {
    if (railLayoutQuery.matches || !id) return;
    documentScrollPositions.set(scrollPositionKey(id, libraryMode), window.scrollY);
  }

  function restoreDocumentScrollPosition(id, libraryMode = isLibraryMode, fallback = null) {
    if (railLayoutQuery.matches || !id) return;
    const key = scrollPositionKey(id, libraryMode);
    const hasSavedPosition = documentScrollPositions.has(key);
    if (!hasSavedPosition && fallback === null) return;
    const top = hasSavedPosition ? documentScrollPositions.get(key) : fallback;
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: Math.max(0, Number(top) || 0), behavior: 'auto' });
    });
  }

  function canonicalPanelHash(id) {
    return `#${encodeURIComponent(id)}`;
  }

  function updateLocation(id, mode = 'push', libraryMode = isLibraryMode) {
    const url = new URL(window.location.href);
    url.hash = id;
    if (libraryMode) {
      url.searchParams.set('view', 'library');
    } else {
      url.searchParams.delete('view');
    }
    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextLocation === currentLocation) return;
    const historyMethod = mode === 'replace' ? 'replaceState' : 'pushState';
    if (typeof window.history?.[historyMethod] === 'function') {
      window.history[historyMethod]({ homePanel: id, homeView: libraryMode ? 'library' : 'overview' }, '', url);
    } else {
      window.location.assign(url.toString());
    }
  }

  function normalizePanelHash(id) {
    if (!id || !window.location.hash || window.location.hash === canonicalPanelHash(id)) return;
    updateLocation(id, 'replace');
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
      if (!scroller) return;
      const independentlyScrollable = railLayoutQuery.matches &&
        id === activeId &&
        !panelById.get(id)?.hidden &&
        scroller.scrollHeight > scroller.clientHeight + 1;
      if (independentlyScrollable) {
        scroller.setAttribute('tabindex', '0');
      } else {
        scroller.removeAttribute('tabindex');
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
    const isNonCollapsible = selected && (triggerId === defaultPanel || isLibraryMode);
    trigger.setAttribute('aria-expanded', String(selected));
    if (isNonCollapsible) {
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

  function markViewTransition() {
    if (viewTransitionTimer) window.clearTimeout(viewTransitionTimer);
    root.classList.add('is-view-changing');
    viewTransitionTimer = window.setTimeout(() => {
      root.classList.remove('is-view-changing');
      viewTransitionTimer = 0;
    }, VIEW_TRANSITION_MS);
  }

  function applyLibraryMode(nextLibraryMode, options = {}) {
    const next = Boolean(nextLibraryMode);
    if (next === isLibraryMode && options.force !== true) {
      updateLibraryViewVisibility();
      return false;
    }

    const apply = () => {
      isLibraryMode = next;
      root.classList.toggle('is-library-mode', next);
      root.dataset.homeView = next ? 'library' : 'overview';
      updateLibraryViewVisibility();
      updateTriggerState(triggerById.get(activeId), true);
      scheduleScrollerTabStopUpdate();
      if (typeof options.afterApply === 'function') options.afterApply();
    };

    if (options.animate !== false && !reducedMotionQuery.matches && typeof document.startViewTransition === 'function') {
      document.startViewTransition(apply);
    } else {
      apply();
      if (options.animate !== false && !reducedMotionQuery.matches) markViewTransition();
    }
    return true;
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

  function activatePanelTrigger(id) {
    if (!ids.includes(id)) return false;
    if (isLibraryMode) return selectPanel(id);
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
    const index = triggers.indexOf(current);
    if (index < 0) return;
    const nextIndex = (index + direction + triggers.length) % triggers.length;
    triggers[nextIndex]?.focus();
  }

  function focusLibraryHeading(id) {
    window.requestAnimationFrame(() => {
      libraryViewById.get(id)?.querySelector('[data-home-library-heading]')?.focus({ preventScroll: true });
    });
  }

  function openLibrary(id) {
    if (!libraryIds.has(id)) return;
    if (id !== activeId) selectPanel(id, { updateHistory: false, reveal: false });
    saveScrollPosition(id, false);
    saveDocumentScrollPosition(id, false);
    renderLibrary(id);
    applyLibraryMode(true);
    updateLocation(id, 'push', true);
    restoreScrollPosition(id, true);
    restoreDocumentScrollPosition(id, true, 0);
    focusLibraryHeading(id);
    root.dispatchEvent(new CustomEvent('home:library-change', {
      bubbles: true,
      detail: { category: id, expanded: true }
    }));
  }

  function closeLibrary(options = {}) {
    if (!isLibraryMode) return;
    const closingId = activeId;
    const restoreFocus = options.restoreFocus !== false
      ? () => {
          window.requestAnimationFrame(() => {
            const returnTarget = root.querySelector(`[data-home-library-open="${closingId}"]`)
              || triggerById.get(closingId);
            returnTarget?.focus({ preventScroll: true });
          });
        }
      : null;
    saveScrollPosition(closingId, true);
    saveDocumentScrollPosition(closingId, true);
    applyLibraryMode(false, { afterApply: restoreFocus });
    updateLocation(closingId, options.historyMode || 'push', false);
    restoreScrollPosition(closingId, false);
    restoreDocumentScrollPosition(closingId, false, 0);
    root.dispatchEvent(new CustomEvent('home:library-change', {
      bubbles: true,
      detail: { category: closingId, expanded: false }
    }));
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
        triggers[0]?.focus();
      } else if (event.key === 'End') {
        event.preventDefault();
        triggers[triggers.length - 1]?.focus();
      } else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activatePanelTrigger(String(trigger.dataset.homeAccordionTrigger || ''));
      }
    });
  });

  libraryOpenButtons.forEach((button) => {
    button.addEventListener('click', () => {
      openLibrary(String(button.dataset.homeLibraryOpen || ''));
    });
  });

  libraryCloseButtons.forEach((button) => {
    button.addEventListener('click', () => closeLibrary());
  });

  function handleLocationChange() {
    const previousLibraryMode = isLibraryMode;
    saveDocumentScrollPosition(activeId, previousLibraryMode);
    const nextLibraryMode = locationRequestsLibrary();
    const hashPanel = panelIdFromHash();
    if (window.location.hash && !hashPanel) return;
    const nextPanel = hashPanel || (nextLibraryMode && ids.includes('projects') ? 'projects' : defaultPanel);
    applyLibraryMode(nextLibraryMode, { animate: true });
    if (hashPanel) normalizePanelHash(hashPanel);
    else updateLocation(nextPanel, 'replace', nextLibraryMode);
    selectPanel(nextPanel, { updateHistory: false, reveal: !nextLibraryMode });
    updateLibraryViewVisibility();
    restoreDocumentScrollPosition(nextPanel, nextLibraryMode, nextLibraryMode ? 0 : null);
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
  listenForMediaChange(reducedMotionQuery, settleClosingPanels);
  window.addEventListener('resize', scheduleScrollerTabStopUpdate);
  root.addEventListener('load', scheduleScrollerTabStopUpdate, true);
  if (typeof window.ResizeObserver === 'function') {
    const scrollerResizeObserver = new window.ResizeObserver(scheduleScrollerTabStopUpdate);
    scrollerById.forEach((scroller) => {
      if (scroller) scrollerResizeObserver.observe(scroller);
    });
  }

  const initialLibraryMode = locationRequestsLibrary();
  const initialHashPanel = panelIdFromHash();
  const initialPanel = initialHashPanel || (initialLibraryMode && ids.includes('projects') ? 'projects' : defaultPanel);
  activeId = initialPanel;
  root.dataset.activePanel = initialPanel;
  scrollerById.forEach((scroller) => scroller?.removeAttribute('tabindex'));
  applyPanelState(initialPanel);
  applyLibraryMode(initialLibraryMode, { animate: false, force: true });
  if (initialHashPanel) {
    normalizePanelHash(initialHashPanel);
    revealPanelTrigger(initialHashPanel);
  } else if (initialLibraryMode) {
    updateLocation(initialPanel, 'replace', true);
  } else if (!window.location.hash) {
    updateLocation(initialPanel, 'replace', false);
  }
  root.dataset.enhanced = 'true';
})();
