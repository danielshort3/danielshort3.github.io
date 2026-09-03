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
  const ids = [...itemById.keys()].filter(Boolean);
  const railLayoutQuery = window.matchMedia('(min-width: 960px) and (min-height: 620px)');
  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scrollPositions = new Map();
  const closeTimers = new Map();
  const LEGACY_LIBRARY_ROUTES = Object.freeze({
    projects: '/portfolio',
    tools: '/tools',
    games: '/games'
  });
  const PANEL_TRANSITION_MS = 520;
  const defaultPanel = ids.includes(root.dataset.defaultPanel)
    ? root.dataset.defaultPanel
    : ids[0];
  let activeId = ids.find((id) => triggerById.get(id)?.getAttribute('aria-expanded') === 'true')
    || defaultPanel;
  let scrollerTabStopFrame = 0;

  function decodedLocationHash() {
    const rawId = String(window.location.hash || '').replace(/^#/, '');
    if (!rawId) return '';
    try {
      return decodeURIComponent(rawId);
    } catch (error) {
      return '';
    }
  }

  function panelIdFromHash() {
    const id = decodedLocationHash();
    return ids.includes(id) ? id : '';
  }

  function normalizeLegacyLibraryLocation() {
    let url;
    try {
      url = new URL(window.location.href);
    } catch (error) {
      return false;
    }
    if (url.searchParams.get('view') !== 'library') return false;

    const legacyRoute = LEGACY_LIBRARY_ROUTES[decodedLocationHash()];
    url.searchParams.delete('view');
    if (legacyRoute) {
      const remainingQuery = url.searchParams.toString();
      window.location.replace(`${legacyRoute}${remainingQuery ? `?${remainingQuery}` : ''}`);
      return true;
    }

    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextLocation !== currentLocation && typeof window.history?.replaceState === 'function') {
      window.history.replaceState(window.history.state, '', url);
    }
    return false;
  }

  if (normalizeLegacyLibraryLocation()) return;

  function panelScrollTarget(id) {
    if (railLayoutQuery.matches && id === 'about') {
      return timelineScrollerById.get(id) || scrollerById.get(id);
    }
    return scrollerById.get(id);
  }

  function panelScrollRegions(id) {
    return [scrollerById.get(id), timelineScrollerById.get(id)].filter(Boolean);
  }

  function saveScrollPosition(id) {
    if (!railLayoutQuery.matches || !id) return;
    const scroller = panelScrollTarget(id);
    if (scroller) scrollPositions.set(id, scroller.scrollTop);
  }

  function restoreScrollPosition(id) {
    if (!railLayoutQuery.matches || !id) return;
    window.requestAnimationFrame(() => {
      const scroller = panelScrollTarget(id);
      if (scroller) scroller.scrollTop = scrollPositions.get(id) || 0;
      scheduleScrollerTabStopUpdate();
    });
  }

  function canonicalPanelHash(id) {
    return `#${encodeURIComponent(id)}`;
  }

  function updateLocation(id, mode = 'push') {
    const url = new URL(window.location.href);
    url.searchParams.delete('view');
    url.hash = id;
    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextLocation === currentLocation) return;
    const historyMethod = mode === 'replace' ? 'replaceState' : 'pushState';
    if (typeof window.history?.[historyMethod] === 'function') {
      window.history[historyMethod]({ homePanel: id }, '', url);
    } else {
      window.location.assign(url.toString());
    }
  }

  function normalizePanelHash(id) {
    if (!id || !window.location.hash || window.location.hash === canonicalPanelHash(id)) return;
    updateLocation(id, 'replace');
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
    if (selected && triggerId === defaultPanel) {
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
    scheduleScrollerTabStopUpdate();
  }

  function revealPanelTrigger(id) {
    if (railLayoutQuery.matches || !id) return;
    window.requestAnimationFrame(() => {
      triggerById.get(id)?.scrollIntoView({
        behavior: reducedMotionQuery.matches ? 'auto' : 'smooth',
        block: 'start'
      });
    });
  }

  function selectPanel(id, options = {}) {
    if (!ids.includes(id)) return false;
    if (id === activeId) return false;

    const previousId = activeId;
    saveScrollPosition(previousId);
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
    if (options.reveal !== false) revealPanelTrigger(id);

    root.dispatchEvent(new CustomEvent('home:category-change', {
      bubbles: true,
      detail: { category: id, view: 'overview' }
    }));
    return true;
  }

  function resolveTriggerTarget(currentId, requestedId, fallbackId) {
    if (requestedId === currentId && requestedId !== fallbackId) return fallbackId;
    return requestedId;
  }

  function activatePanelTrigger(id) {
    if (!ids.includes(id)) return false;
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

  function handleLocationChange() {
    if (normalizeLegacyLibraryLocation()) return;
    const hashPanel = panelIdFromHash();
    if (window.location.hash && !hashPanel) return;
    const nextPanel = hashPanel || defaultPanel;
    if (hashPanel) normalizePanelHash(hashPanel);
    else updateLocation(nextPanel, 'replace');
    selectPanel(nextPanel, { updateHistory: false, reveal: true });
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
    ids.forEach((id) => {
      panelScrollRegions(id).forEach((region) => scrollerResizeObserver.observe(region));
    });
  }

  const initialHashPanel = panelIdFromHash();
  const initialPanel = initialHashPanel || defaultPanel;
  activeId = initialPanel;
  root.dataset.activePanel = initialPanel;
  ids.forEach((id) => panelScrollRegions(id).forEach((region) => region.removeAttribute('tabindex')));
  applyPanelState(initialPanel);
  if (initialHashPanel) {
    normalizePanelHash(initialHashPanel);
    revealPanelTrigger(initialHashPanel);
  } else if (!window.location.hash) {
    updateLocation(initialPanel, 'replace');
  }
  root.dataset.enhanced = 'true';
})();
