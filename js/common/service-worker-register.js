(function () {
  'use strict';

  if (!('serviceWorker' in navigator) || !window.isSecureContext) return;
  if (new URLSearchParams(window.location.search).has('no_sw')) return;
  const host = String(window.location.hostname || '').toLowerCase();
  if (!host || host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' || host === '::1' || /^(\d{1,3}\.){3}\d{1,3}$/.test(host)) return;

  function registerServiceWorker() {
    navigator.serviceWorker.register('/sw.js').catch(function () {});
  }

  if (document.readyState === 'complete') {
    registerServiceWorker();
  } else {
    window.addEventListener('load', registerServiceWorker, { once: true });
  }
})();
