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
  const triggerById = new Map(triggers.map((trigger) => [
    String(trigger.dataset.homeAccordionTrigger || ''),
    trigger
  ]));
  const itemById = new Map(items.map((item) => [
    String(item.dataset.homeAccordionItem || ''),
    item
  ]));
  const ids = [...itemById.keys()].filter(Boolean);
  const desktopQuery = window.matchMedia('(min-width: 769px)');
  const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scrollPositions = new Map();
  const defaultPanel = ids.includes(root.dataset.defaultPanel)
    ? root.dataset.defaultPanel
    : ids[0];
  let activeId = ids.find((id) => triggerById.get(id)?.getAttribute('aria-expanded') === 'true')
    || defaultPanel;

  function panelIdFromHash() {
    const id = decodeURIComponent(String(window.location.hash || '').replace(/^#/, ''));
    return ids.includes(id) ? id : '';
  }

  function saveScrollPosition(id) {
    if (!desktopQuery.matches || !id) return;
    const scroller = panelById.get(id)?.querySelector('[data-home-accordion-scroller]');
    if (scroller) scrollPositions.set(id, scroller.scrollTop);
  }

  function restoreScrollPosition(id) {
    if (!desktopQuery.matches || !id) return;
    window.requestAnimationFrame(() => {
      const scroller = panelById.get(id)?.querySelector('[data-home-accordion-scroller]');
      if (scroller) scroller.scrollTop = scrollPositions.get(id) || 0;
    });
  }

  function updateHash(id) {
    if (!window.history?.replaceState || window.location.hash === `#${id}`) return;
    const url = new URL(window.location.href);
    url.hash = id;
    window.history.replaceState(window.history.state, '', url);
  }

  function selectPanel(id, options = {}) {
    if (!ids.includes(id) || id === activeId) return false;

    saveScrollPosition(activeId);
    activeId = id;
    root.dataset.activePanel = id;

    items.forEach((item) => {
      const itemId = String(item.dataset.homeAccordionItem || '');
      const selected = itemId === id;
      const trigger = triggerById.get(itemId);
      const panel = panelById.get(itemId);
      item.classList.toggle('is-active', selected);
      trigger?.setAttribute('aria-expanded', String(selected));
      if (panel) {
        panel.hidden = !selected;
        if (selected) {
          panel.removeAttribute('inert');
        } else {
          panel.setAttribute('inert', '');
        }
      }
    });

    if (options.updateHash !== false) updateHash(id);
    restoreScrollPosition(id);

    if (!desktopQuery.matches && options.reveal !== false) {
      window.requestAnimationFrame(() => {
        triggerById.get(id)?.scrollIntoView({
          behavior: reducedMotionQuery.matches ? 'auto' : 'smooth',
          block: 'start'
        });
      });
    }

    root.dispatchEvent(new CustomEvent('home:category-change', {
      bubbles: true,
      detail: { category: id }
    }));
    return true;
  }

  function focusRelativeTrigger(current, direction) {
    const index = triggers.indexOf(current);
    if (index < 0) return;
    const nextIndex = (index + direction + triggers.length) % triggers.length;
    triggers[nextIndex]?.focus();
  }

  triggers.forEach((trigger) => {
    trigger.addEventListener('click', () => {
      selectPanel(String(trigger.dataset.homeAccordionTrigger || ''));
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
        selectPanel(String(trigger.dataset.homeAccordionTrigger || ''));
      }
    });
  });

  window.addEventListener('hashchange', () => {
    const hashPanel = panelIdFromHash();
    if (hashPanel) selectPanel(hashPanel, { updateHash: false });
  });

  const initialPanel = panelIdFromHash() || activeId || defaultPanel;
  if (initialPanel !== activeId) {
    selectPanel(initialPanel, { updateHash: false, reveal: false });
  } else {
    root.dataset.activePanel = initialPanel;
  }
  root.dataset.enhanced = 'true';
})();
