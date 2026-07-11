/* ===================================================================
   File: ga4-events.js
   Purpose: Consent-aware event helpers for the GTM-managed GA4 tag
   =================================================================== */
(() => {
  'use strict';

  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function(){ (window.dataLayer = window.dataLayer || []).push(arguments); };

  let analyticsConsentGranted = false;
  let domListenersBound = false;
  let contactPageViewSent = false;
  let engagementTimer = null;
  let engagementSent = false;
  const viewedProjectIds = new Set();
  let sent50 = false;

  function normalizeConsentState(value) {
    return value && value.categories ? value.categories : value;
  }

  function isEmbeddedSameOrigin() {
    try {
      if (window.self === window.top) return false;
      return window.top.location.origin === window.location.origin;
    } catch (err) {
      return false;
    }
  }

  function readConsentState() {
    if (!window.consentAPI || typeof window.consentAPI.get !== 'function') return null;
    try {
      return normalizeConsentState(window.consentAPI.get());
    } catch (err) {
      return null;
    }
  }

  function send(name, params = {}) {
    const eventName = String(name || '').trim();
    if (!analyticsConsentGranted || !eventName) return false;
    window.gtag('event', eventName, params && typeof params === 'object' ? params : {});
    return true;
  }

  function trackContactPageView() {
    if (contactPageViewSent || !document.body || document.body.dataset.page !== 'contact') return;
    contactPageViewSent = send('contact_page_view');
  }

  function stopEngagementTimer() {
    if (engagementTimer === null) return;
    clearTimeout(engagementTimer);
    engagementTimer = null;
  }

  function startEngagementTimer() {
    if (!analyticsConsentGranted || engagementSent || engagementTimer !== null) return;
    engagementTimer = setTimeout(() => {
      engagementTimer = null;
      engagementSent = send('engaged_time', { seconds: 60 });
    }, 60000);
  }

  function updateConsent(value) {
    const state = normalizeConsentState(value);
    analyticsConsentGranted = !isEmbeddedSameOrigin() && !!(state && state.analytics);
    if (!analyticsConsentGranted) {
      stopEngagementTimer();
      return;
    }
    trackContactPageView();
    startEngagementTimer();
  }

  function getClosest(target, selector) {
    return target && typeof target.closest === 'function' ? target.closest(selector) : null;
  }

  function getResumeDetails(link) {
    const href = link && typeof link.getAttribute === 'function' ? link.getAttribute('href') : '';
    if (!href) return null;
    try {
      const url = new URL(href, document.baseURI || window.location.href);
      const fileName = decodeURIComponent(url.pathname.split('/').pop() || '');
      if (!/^Resume(?:-[A-Za-z0-9-]+)?\.pdf$/i.test(fileName)) return null;
      const explicitVariant = fileName
        .replace(/^Resume-?/i, '')
        .replace(/\.pdf$/i, '')
        .trim()
        .toLowerCase();
      const pageVariant = document.body && document.body.dataset
        ? String(document.body.dataset.audience || '').trim().toLowerCase()
        : '';
      return {
        fileName,
        linkUrl: url.href,
        variant: explicitVariant || pageVariant || 'general'
      };
    } catch (err) {
      return null;
    }
  }

  function getOutboundDetails(link) {
    const href = link && typeof link.getAttribute === 'function' ? link.getAttribute('href') : '';
    if (!href) return null;
    try {
      const url = new URL(href, document.baseURI || window.location.href);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
      if (url.hostname === window.location.hostname) return null;
      return {
        linkUrl: url.href,
        linkDomain: url.hostname
      };
    } catch (err) {
      return null;
    }
  }

  function handleDocumentClick(event) {
    const target = event && event.target;
    const link = getClosest(target, 'a[href]');
    const heroCta = getClosest(target, '.hero-cta');
    if (heroCta) {
      send('hero_cta_click', { cta_label: String(heroCta.textContent || '').trim() });
    }

    const filterOption = getClosest(target, '#filter-menu [data-filter]');
    if (filterOption && filterOption.dataset) {
      send('project_filter_select', { filter_name: filterOption.dataset.filter });
    }

    const seeMore = getClosest(target, '#see-more');
    if (seeMore && seeMore.dataset) {
      send('see_more_toggle', { expanded: seeMore.dataset.expanded !== 'true' });
    }

    if (!link) return;

    const href = String(link.getAttribute('href') || '').trim();
    const linkText = String(link.textContent || '').trim();
    const resume = getResumeDetails(link);
    if (resume) {
      send('resume_download', {
        file_name: resume.fileName,
        resume_variant: resume.variant,
        link_url: resume.linkUrl
      });
    }

    if (/^mailto:/i.test(href)) {
      send('email_cta_click', { link_url: href });
    }

    const outbound = getOutboundDetails(link);
    if (outbound) {
      send('outbound_click', {
        link_url: outbound.linkUrl,
        link_domain: outbound.linkDomain,
        link_text: linkText
      });
    }

    if (link.classList && link.classList.contains('nav-link') && link.getAttribute('target') !== '_blank') {
      send('nav_link_click', { link_url: href });
    }

    const contactCard = getClosest(link, '.contact-card');
    if (document.body && document.body.dataset.page === 'contact' && contactCard) {
      send('contact_card_click', { card_label: linkText || href });
    }
  }

  function handleScroll() {
    if (sent50 || !analyticsConsentGranted) return;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    if (scrollable <= 0) return;
    const pct = (window.scrollY || window.pageYOffset || 0) / scrollable;
    if (pct >= 0.5) sent50 = send('scroll_depth', { percent: 50 });
  }

  function bindDomListeners() {
    if (domListenersBound) return;
    domListenersBound = true;
    document.addEventListener('click', handleDocumentClick);
    window.addEventListener('scroll', handleScroll, { passive: true });
    trackContactPageView();
  }

  window.gaEvent = send;

  window.trackProjectView = id => {
    const projectId = String(id || '').trim();
    if (!projectId || !send('project_view', { project_id: projectId })) return false;
    const previousCount = viewedProjectIds.size;
    viewedProjectIds.add(projectId.toLowerCase());
    if (previousCount < 3 && viewedProjectIds.size === 3) {
      send('multi_project_view', { view_count: 3 });
    }
    return true;
  };

  window.trackModalClose = id => send('modal_close', { project_id: id });

  window.addEventListener('consent-changed', event => updateConsent(event.detail));

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindDomListeners, { once: true });
  } else {
    bindDomListeners();
  }

  updateConsent(readConsentState());
})();
