(function () {
  'use strict';

  function mountTableauControls(root = document) {
    const cleanups = [];
    root.querySelectorAll('[data-dashboard-reset]').forEach((button) => {
      if (button.dataset.dashboardResetReady) return;
      const shell = button.closest('.project-demo-shell');
      const frame = shell?.querySelector('iframe[data-dashboard-default-src]');
      const defaultSrc = frame?.getAttribute('data-dashboard-default-src');
      if (!defaultSrc) return;
      // A fresh native view also clears chart selections, navigation and map zoom.
      // Retain the original URL rather than a URL changed by an interactive view.
      const reset = () => {
        const resetUrl = new URL(defaultSrc, document.baseURI);
        resetUrl.searchParams.set(':revert', 'all');
        resetUrl.searchParams.delete(':iid');
        const device = frame.getAttribute('data-dashboard-device');
        if (device === 'phone' || device === 'desktop') resetUrl.searchParams.set(':device', device);
        frame.removeAttribute('data-src');
        frame.setAttribute('src', resetUrl.href);
      };
      button.addEventListener('click', reset);
      button.dataset.dashboardResetReady = 'true';
      button.hidden = false;
      cleanups.push(() => {
        button.removeEventListener('click', reset);
        delete button.dataset.dashboardResetReady;
        button.hidden = true;
      });
    });
    return () => cleanups.forEach((cleanup) => cleanup());
  }

  window.TableauControls = Object.freeze({ mount: mountTableauControls });
  const mount = (root) => {
    const cleanup = mountTableauControls(root);
    window.SiteRoutes?.addCleanup?.(cleanup);
  };
  document.addEventListener('DOMContentLoaded', () => mount(document));
  document.addEventListener('site:content-updated', (event) => mount(event.detail?.root || document));
  if (document.readyState !== 'loading') {
    mount(document.querySelector('[data-site-route-content], [data-personal-detail-content]') || document);
  }
})();
