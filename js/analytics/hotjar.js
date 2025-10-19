/* Load Hotjar only after analytics consent is granted */
(() => {
  'use strict';

  const vendorConfig = (window.PrivacyConfig && window.PrivacyConfig.vendors && window.PrivacyConfig.vendors.hotjar) || {};
  const attrId = document.documentElement.getAttribute('data-hotjar-id');
  const siteId = vendorConfig.id || attrId;
  const scriptVersion = vendorConfig.version || 6;

  if (!siteId || String(siteId).trim() === '') return;

  const loadHotjar = () => {
    if (window.hj) return;
    (function(h,o,t,j,a,r){
      h.hj = h.hj || function(){ (h.hj.q = h.hj.q || []).push(arguments); };
      h._hjSettings = { hjid: Number(siteId), hjsv: scriptVersion };
      a = o.getElementsByTagName('head')[0];
      r = o.createElement('script'); r.async = 1;
      r.src = `https://static.hotjar.com/c/hotjar-${siteId}.js?sv=${scriptVersion}`;
      a.appendChild(r);
    })(window, document);
  };

  const shouldLoad = (detail) => {
    const state = detail && detail.categories ? detail.categories : detail;
    return !!(state && state.analytics);
  };

  if (window.consentAPI && typeof window.consentAPI.get === 'function') {
    const state = window.consentAPI.get();
    if (shouldLoad(state)) loadHotjar();
  }

  window.addEventListener('consent-changed', (evt) => {
    if (shouldLoad(evt.detail)) loadHotjar();
  });
})();
