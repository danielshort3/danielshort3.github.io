/* Shared, route-safe tabs for the tool workspaces. */
(() => {
  'use strict';

  if (window.ToolWorkspace) {
    window.ToolWorkspace.init(document);
    return;
  }

  const ownElements = (group, selector) => Array.from(group.querySelectorAll(selector))
    .filter((element) => element.closest('[data-workspace-tabset]') === group);

  const selectTab = (panelId, options = {}) => {
    const panel = document.getElementById(panelId);
    const group = panel?.closest('[data-workspace-tabset]');
    if (!group) return false;
    const tabs = ownElements(group, '[data-workspace-tab]');
    const selected = tabs.find((tab) => tab.getAttribute('aria-controls') === panelId);
    if (!selected || selected.disabled || selected.hidden) return false;
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    ownElements(group, '[data-workspace-panel]').forEach((item) => {
      item.hidden = item.id !== panelId;
    });
    if (options.focus) selected.focus({ preventScroll: true });
    if (options.notify !== false) {
      group.dispatchEvent(new CustomEvent('tool:tab-change', {
        bubbles: true,
        detail: { panelId, tab: selected, panel }
      }));
    }
    return true;
  };

  const init = (root = document) => {
    const groups = Array.from(root.querySelectorAll('[data-workspace-tabset]'));
    if (root.matches?.('[data-workspace-tabset]')) groups.unshift(root);
    groups.forEach((group) => {
      const tabs = ownElements(group, '[data-workspace-tab]');
      const selected = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true' && !tab.disabled && !tab.hidden)
        || tabs.find((tab) => !tab.disabled && !tab.hidden);
      if (selected) selectTab(selected.getAttribute('aria-controls'), { notify: false });
    });
  };

  document.addEventListener('click', (event) => {
    const tab = event.target.closest?.('[data-workspace-tab]');
    if (!tab || !tab.closest('[data-workspace-tabset]')) return;
    event.preventDefault();
    selectTab(tab.getAttribute('aria-controls'));
  });

  document.addEventListener('keydown', (event) => {
    const tab = event.target.closest?.('[data-workspace-tab]');
    const group = tab?.closest('[data-workspace-tabset]');
    if (!group) return;
    const tabs = ownElements(group, '[data-workspace-tab]').filter((item) => !item.disabled && !item.hidden);
    const index = tabs.indexOf(tab);
    const vertical = tab.closest('[role="tablist"]')?.getAttribute('aria-orientation') === 'vertical';
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else if (event.key === (vertical ? 'ArrowDown' : 'ArrowRight')) next = (index + 1) % tabs.length;
    else if (event.key === (vertical ? 'ArrowUp' : 'ArrowLeft')) next = (index - 1 + tabs.length) % tabs.length;
    else return;
    event.preventDefault();
    if (tabs[next]) selectTab(tabs[next].getAttribute('aria-controls'), { focus: true });
  });

  const api = { init, selectTab };
  window.ToolWorkspace = api;
  window.SiteRoutes?.addCleanup?.(() => {
    if (window.ToolWorkspace === api) delete window.ToolWorkspace;
  });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => init(), { once: true });
  } else init();
  document.addEventListener('site:route-mounted', (event) => init(event.detail?.root || document));
})();
