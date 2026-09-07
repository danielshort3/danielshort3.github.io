(function () {
  'use strict';

  const DEBUG_STORAGE_KEY = 'site_analytics_debug_v1';
  const PRODUCTION_HOSTS = ['www.danielshort.me', 'danielshort.me'];
  let isProduction = false;
  let debug = false;
  try {
    const url = new URL(window.location.href);
    isProduction = url.protocol === 'https:' && PRODUCTION_HOSTS.includes(url.hostname);
    const requestedDebug = url.searchParams.get('analytics_debug');
    const isQaCampaign = String(url.searchParams.get('utm_source') || '').toLowerCase() === 'qa';
    let savedDebug = false;
    try {
      if (requestedDebug === '0') window.sessionStorage.removeItem(DEBUG_STORAGE_KEY);
      savedDebug = window.sessionStorage.getItem(DEBUG_STORAGE_KEY) === '1';
    } catch (err) {}
    debug = requestedDebug === '1' || isQaCampaign || savedDebug;
    try {
      if (debug) window.sessionStorage.setItem(DEBUG_STORAGE_KEY, '1');
    } catch (err) {}
  } catch (err) {}

  window.SiteAnalyticsEnvironment = Object.freeze({
    enabled: isProduction || debug,
    debug,
    trafficType: debug ? 'internal' : undefined
  });
  window.dataLayer = window.dataLayer || [];
  // GA4 treats even debug_mode=false as debug traffic. Omit it for visitors.
  window.dataLayer.push({
    analytics_debug: debug ? true : undefined,
    traffic_type: debug ? 'internal' : undefined
  });
})();
