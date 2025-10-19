(() => {
  'use strict';
  const body = document.body;
  if (!body) return;
  const configSite = (window.PrivacyConfig && window.PrivacyConfig.vendors && window.PrivacyConfig.vendors.hotjar && window.PrivacyConfig.vendors.hotjar.id) || '';
  const siteAttr = body.dataset.hotjarSite || String(configSite || '');
  const siteId = parseInt(siteAttr, 10);
  if (!siteId) return;
  if (window.hj) return;

  window.hj = window.hj || function(){ (window.hj.q = window.hj.q || []).push(arguments); };
  window._hjSettings = {
    hjid: siteId,
    hjsv: parseInt(body.dataset.hotjarVersion || '6', 10) || 6
  };
  const script = document.createElement('script');
  script.async = true;
  script.src = `https://static.hotjar.com/c/hotjar-${window._hjSettings.hjid}.js?sv=${window._hjSettings.hjsv}`;
  document.head.appendChild(script);
})();
