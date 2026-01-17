(() => {
  'use strict';

  const normalizePath = (value) => (
    String(value || '')
      .trim()
      .replace(/^\/+/, '')
      .replace(/\/+$/, '')
  );

  try {
    const url = new URL(window.location.href);
    const from = normalizePath(url.searchParams.get('from')) || 'dshort.me';
    const path = normalizePath(url.searchParams.get('path'));
    const requested = path ? `https://${from}/${path}` : `https://${from}/`;
    const requestedEl = document.querySelector('[data-dshort="requested"]');
    if (requestedEl) requestedEl.textContent = requested;
  } catch (_) {}
})();

