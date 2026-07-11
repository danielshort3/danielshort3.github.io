/*
 * consent_manager.js
 *
 * A lightweight, privacy‑focused consent management implementation for
 * static websites. It displays an accessible banner and preferences
 * modal, persists the user’s choices, integrates with Google Consent
 * Mode v2 and optional IAB TCF v2.2 generation, and honors
 * Global Privacy Control (GPC) signals for California residents.
 *
 * Google Tag Manager is loaded only after analytics consent is granted.
 * TCF generation is disabled by default and requires both @iabtcf/core
 * and explicit registered CMP metadata in PrivacyConfig.
 */

(function () {
  'use strict';

  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function(){ (window.dataLayer = window.dataLayer || []).push(arguments); };
  try {
    window.gtag('consent', 'default', {
      ad_storage: 'denied',
      analytics_storage: 'denied',
      ad_user_data: 'denied',
      ad_personalization: 'denied',
      wait_for_update: 500
    });
  } catch (err) {}

  const STYLE_ID = 'pcz-consent-styles';
  const CRITICAL_STYLE_ID = 'pcz-consent-critical-styles';
  const CSS_VERSION = 'v9';

  function loadStyles() {
    if (!document.getElementById(CRITICAL_STYLE_ID)) {
      const critical = document.createElement('style');
      critical.id = CRITICAL_STYLE_ID;
      critical.textContent = [
        '#pcz-banner,#pcz-modal{font:16px/1.5 "Inter","Segoe UI",Arial,sans-serif;color:#091f3b;}',
        '#pcz-banner .pcz-card,#pcz-modal .pcz-panel{background:#fff;color:#091f3b;border:1px solid rgba(9,31,59,.14);box-shadow:0 10px 26px rgba(9,31,59,.1);border-radius:10px;}',
        '#pcz-banner .pcz-body,#pcz-modal .policy-lead,#pcz-modal .pref-description{color:#334155;}',
        '#pcz-banner .pcz-primary,#pcz-modal .pcz-save-preferences{background:#005fed!important;color:#fff!important;border-color:#005fed!important;}',
        '#pcz-banner .pcz-secondary{background:#fff!important;color:#091f3b!important;border-color:rgba(9,31,59,.2)!important;}',
        '#pcz-banner .pcz-link{color:#005fed!important;}',
        '#pcz-banner .pcz-close,#pcz-modal .pcz-panel-close{background:#fff;color:#334155;border-color:rgba(9,31,59,.16);}'
      ].join('\n');
      document.head.appendChild(critical);
    }
    if (document.getElementById(STYLE_ID) || document.querySelector('link[href$="privacy.css"]')) return;
    const link = document.createElement('link');
    link.id = STYLE_ID;
    link.rel = 'stylesheet';
    link.href = 'css/privacy.css' + (CSS_VERSION ? ('?v=' + CSS_VERSION) : '');
    document.head.appendChild(link);
  }

  /**
   * Configuration for the CMP. You can extend this object with
   * additional languages or categories as needed. Titles and
   * descriptions should be plain language with no dark patterns.
   */
  const CONFIG = {
    version: 1,
    languages: {
      en: {
        bannerTitle: 'I value your privacy.',
        bannerDesc: 'I use optional cookies to understand traffic and improve the site. Choose the level you are comfortable with.',
        acceptAll: 'Allow all',
        rejectAll: 'Essential only',
        managePrefs: 'Manage settings',
        privacyPolicy: 'Privacy Policy',
        close: 'Close banner and use essential cookies only',
        modalTitle: 'Manage Your Privacy Settings',
        modalLead: 'Adjust your preferences below and click “Save preferences” to apply the changes. Necessary cookies are always enabled.',
        savePrefs: 'Save preferences',
        cancel: 'Cancel',
        closePrefs: 'Close',
        stateOn: 'On',
        stateOff: 'Off',
        stateAlwaysOn: 'Always on',
        requiredLabel: 'Required for site operation',
        categories: {
          necessary: {
            label: 'Strictly necessary',
            description: 'These cookies are required for the site to run: navigation, basic interactions, and honoring your privacy choices. They cannot be disabled.'
          },
          analytics: {
            label: 'Analytics',
            description: 'Analytics cookies help me understand which pages are viewed most often and how visitors move around the site so I can improve the experience.'
          },
          functional: {
            label: 'Functional',
            description: 'Functional cookies remember your settings (like language or filters) so features feel more tailored to you.'
          },
          advertising: {
            label: 'Advertising',
            description: 'Advertising cookies make it possible to personalize or measure marketing efforts. They are only used if you opt in.'
          }
        },
        doNotSell: 'Do Not Sell/Share My Personal Information',
        gpcHonoured: 'Your browser sent a Global Privacy Control signal so we\'ve applied your opt‑out. You can adjust other preferences below.'
      },
      es: {
        bannerTitle: 'Valoro tu privacidad',
        bannerDesc: 'Utilizo cookies opcionales para entender el tráfico y mejorar el sitio. Elige el nivel con el que te sientas cómodo.',
        acceptAll: 'Permitir todas',
        rejectAll: 'Solo esenciales',
        managePrefs: 'Administrar ajustes',
        privacyPolicy: 'Política de privacidad',
        close: 'Cerrar y usar solo cookies esenciales',
        modalTitle: 'Preferencias de privacidad',
        modalLead: 'Ajusta tus preferencias y pulsa “Guardar preferencias” para aplicar los cambios. Las cookies necesarias siempre están activadas.',
        savePrefs: 'Guardar preferencias',
        cancel: 'Cancelar',
        closePrefs: 'Cerrar',
        stateOn: 'Sí',
        stateOff: 'No',
        stateAlwaysOn: 'Siempre activas',
        requiredLabel: 'Necesarias para el funcionamiento del sitio',
        categories: {
          necessary: {
            label: 'Estrictamente necesarias',
            description: 'Estas cookies son necesarias para funciones básicas del sitio web como la navegación y el acceso a áreas seguras. No se pueden desactivar.'
          },
          analytics: {
            label: 'Analíticas',
            description: 'Las cookies de análisis nos ayudan a entender cómo interactúan los visitantes con nuestro sitio y mejorar su rendimiento.'
          },
          functional: {
            label: 'Funcionales',
            description: 'Las cookies funcionales permiten que el sitio recuerde sus elecciones y ofrezca funciones mejoradas y más personales.'
          },
          advertising: {
            label: 'Publicidad',
            description: 'Las cookies de publicidad se usan para ofrecer anuncios relevantes y medir la eficacia de las campañas de marketing.'
          }
        },
        doNotSell: 'No vender/compartir mi información personal',
        gpcHonoured: 'Su navegador envió una señal Global Privacy Control, así que hemos aplicado su exclusión. Puede ajustar otras preferencias a continuación.'
      }
    },
    vendors: {
      gtm: { enabled: false }
    }
  };

  const GLOBAL_CONF = window.PrivacyConfig || {};
  if (GLOBAL_CONF.vendors) {
    Object.keys(GLOBAL_CONF.vendors).forEach(function (v) {
      CONFIG.vendors[v] = Object.assign(CONFIG.vendors[v] || {}, GLOBAL_CONF.vendors[v]);
    });
  }
  const STORAGE_KEY = GLOBAL_CONF.storageKey || 'consent';

  /**
   * Retrieve the user’s locale. We fall back to English if no
   * supported language is detected.
   */
  function getLocale() {
    const langs = navigator.languages || [navigator.language || 'en'];
    const code = (langs[0] || 'en').split('-')[0].toLowerCase();
    return CONFIG.languages[code] ? code : 'en';
  }

  /**
   * Check if the browser has Global Privacy Control (GPC) enabled.
   */
  function hasGPC() {
    try {
      return navigator.globalPrivacyControl === true;
    } catch (err) {
      return false;
    }
  }

  function hasDNT() {
    try {
      const value = navigator.doNotTrack || window.doNotTrack || navigator.msDoNotTrack;
      return value === '1' || value === 1 || String(value || '').toLowerCase() === 'yes';
    } catch (err) {
      return false;
    }
  }

  function isEmbeddedSameOrigin() {
    try {
      if (window.self === window.top) return false;
      return window.top.location.origin === window.location.origin;
    } catch (err) {
      return false;
    }
  }

  /**
   * Simple region detection. For proper enforcement you should integrate
   * a server‑side IP geolocation service that returns the visitor’s region.
   * This stub always returns 'US'.
   */
  function normalizeRegion(value) {
    if (!value) return '';
    return String(value).trim().toUpperCase();
  }

  function getRegion() {
    const configured = normalizeRegion(GLOBAL_CONF.region || (typeof window !== 'undefined' ? window.PrivacyRegion : ''));
    if (configured) return configured;
    try {
      const nav = navigator || {};
      const lang = (nav.languages && nav.languages[0]) || nav.language || '';
      const lower = String(lang).toLowerCase();
      if (!lower) return 'US';
      if (lower.startsWith('en-gb')) return 'UK';
      const euPrefixes = ['de', 'fr', 'es', 'it', 'pt', 'pl', 'cs', 'da', 'nl', 'sv', 'fi', 'no', 'sk', 'sl', 'ro', 'hu', 'bg', 'hr', 'lt', 'lv', 'et', 'el', 'ga', 'mt'];
      if (euPrefixes.some(code => lower.startsWith(code))) return 'EU';
    } catch {}
    return 'US';
  }

  function getUSState() {
    const configured = normalizeRegion(GLOBAL_CONF.usState || (typeof window !== 'undefined' ? window.PrivacyUSState : ''));
    if (configured.startsWith('US-')) return configured.replace('US-', '');
    if (configured.length === 2) return configured;
    return '';
  }

  function regionRequiresOptIn(region, stateCode) {
    const regionsCfg = GLOBAL_CONF.regions || {};
    const ccpaStates = new Set((regionsCfg.usStatesWithCCPA || []).map(function (code) {
      return String(code || '').trim().toUpperCase();
    }));
    const upperRegion = normalizeRegion(region);
    if (upperRegion === 'EU' || upperRegion === 'EEA') return regionsCfg.eea !== false;
    if (upperRegion === 'UK' || upperRegion === 'UNITED KINGDOM') return regionsCfg.uk !== false;
    if (upperRegion.startsWith('US-')) {
      const state = upperRegion.slice(3);
      if (ccpaStates.has(state)) return true;
      return false;
    }
    if (upperRegion === 'US') {
      return !!(stateCode && ccpaStates.has(stateCode));
    }
    return false;
  }

  function enforcePrivacySignals(state) {
    const result = Object.assign(
      { necessary: true, analytics: false, functional: false, advertising: false },
      state || {},
      { necessary: true }
    );
    if (GLOBAL_CONF.respectDNT !== false && hasDNT()) {
      result.analytics = false;
      result.advertising = false;
    }
    if (GLOBAL_CONF.respectGPC !== false && hasGPC()) result.advertising = false;
    return result;
  }

  function getDefaultState() {
    const strictState = { necessary: true, analytics: false, functional: false, advertising: false };
    return enforcePrivacySignals(strictState);
  }

  function setBannerUiState(isOpen) {
    try {
      if (!document.body) return;
      if (isOpen) {
        document.body.setAttribute('data-consent-banner', 'open');
        return;
      }
      document.body.removeAttribute('data-consent-banner');
    } catch (err) {}
  }

  const isOnline = () => {
    try {
      if (navigator?.onLine === false) return false;
    } catch (err) {}
    return true;
  };

  function getEnabledTcfConfig() {
    const tcfConfig = GLOBAL_CONF.tcf || {};
    const cmpId = Number(tcfConfig.cmpId);
    const cmpVersion = Number(tcfConfig.cmpVersion);
    if (tcfConfig.enabled !== true) return null;
    if (!Number.isInteger(cmpId) || cmpId <= 0) return null;
    if (!Number.isInteger(cmpVersion) || cmpVersion <= 0) return null;
    return { cmpId, cmpVersion };
  }

  /**
   * Persist the consent record in localStorage. The record includes
   * timestamp, region, version, user categories, privacy signals and an
   * optional TCF string when explicitly configured.
   */
  function saveConsent(state) {
    const effectiveState = enforcePrivacySignals(state);
    const record = {
      version: CONFIG.version,
      timestamp: Date.now(),
      region: getRegion(),
      gpc: hasGPC(),
      dnt: hasDNT(),
      categories: effectiveState,
      tcString: ''
    };
    if (record.region === 'EU' && getEnabledTcfConfig() && window.__iabtcf) {
      getTcString(effectiveState).then(tc => {
        record.tcString = tc;
        localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
      });
      return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
  }

  /**
   * Load a previously saved consent record from localStorage.
   */
  function loadConsent() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const record = raw ? JSON.parse(raw) : null;
      if (record && record.categories) record.categories = enforcePrivacySignals(record.categories);
      return record;
    } catch (err) {
      return null;
    }
  }

  /**
   * Apply the chosen consent state by updating Google Consent Mode
   * and enabling or disabling vendors.
   */
  function applyConsent(state) {
    state = enforcePrivacySignals(state);
    const update = {
      analytics_storage: state.analytics ? 'granted' : 'denied',
      ad_storage: state.advertising ? 'granted' : 'denied',
      ad_user_data: state.advertising ? 'granted' : 'denied',
      ad_personalization: state.advertising ? 'granted' : 'denied'
    };
    if (typeof window.gtag === 'function') {
      window.gtag('consent', 'update', update);
    } else {
      window.dataLayer = window.dataLayer || [];
      window.dataLayer.push(function () {
        window.gtag('consent', 'update', update);
      });
    }
    enableVendor('gtm', state.analytics);
    window.dispatchEvent(new CustomEvent('consent-changed', { detail: state }));
  }

  /**
   * Dynamically load or unload a vendor script based on consent.
   */
  function enableVendor(vendorKey, enabled) {
    const vendor = CONFIG.vendors[vendorKey];
    if (!vendor) return;
    vendor._consentGranted = !!enabled;
    if (enabled && !vendor.enabled) {
      if (vendorKey === 'gtm') {
        const containerId = String(vendor.id || '').trim();
        if (!/^GTM-[A-Z0-9]+$/i.test(containerId)) {
          console.warn('Google Tag Manager container ID is missing or invalid.');
          return;
        }
        const scriptId = 'gtm-src';
        const loadGtm = () => {
          if (document.getElementById(scriptId)) return;
          window.dataLayer = window.dataLayer || [];
          window.dataLayer.push({
            'gtm.start': Date.now(),
            event: 'gtm.js'
          });
          const s = document.createElement('script');
          s.id = scriptId;
          s.async = true;
          s.src = 'https://www.googletagmanager.com/gtm.js?id=' + encodeURIComponent(containerId);
          s.onerror = function(){
            vendor.enabled = false;
          };
          document.head.appendChild(s);
        };
        if (!isOnline()) {
          if (!vendor._pendingOnline) {
            vendor._pendingOnline = true;
            window.addEventListener('online', function handleOnline(){
              vendor._pendingOnline = false;
              window.removeEventListener('online', handleOnline);
              if (vendor._consentGranted) enableVendor(vendorKey, true);
            });
          }
          return;
        }
        loadGtm();
      }
      vendor.enabled = true;
    }
    if (!enabled && vendor.enabled) {
      // GTM cannot be fully unloaded after consent is revoked. Consent Mode
      // blocks Google tags, and event helpers independently enforce consent.
      vendor.enabled = false;
    }
  }

  /**
   * Generate a TCF v2.2 compliant string using the @iabtcf/core library.
   */
  async function getTcString(state) {
    const tcfConfig = getEnabledTcfConfig();
    if (!tcfConfig) return '';
    const api = window.__iabtcf;
    if (!api) return '';
    const { TCModel, TCString, GVL } = api;
    const gvl = new GVL();
    await gvl.readyPromise;
    const tcModel = new TCModel(gvl);
    tcModel.cmpId = tcfConfig.cmpId;
    tcModel.cmpVersion = tcfConfig.cmpVersion;
    tcModel.tcStringVersion = 3;
    tcModel.policyVersion = gvl.tcfPolicyVersion;
    tcModel.purposeConsents.set(1, true);
    tcModel.purposeConsents.set(7, state.analytics);
    tcModel.purposeConsents.set(2, state.advertising);
    tcModel.purposeConsents.set(4, state.advertising);
    const allowGoogle = state.analytics || state.advertising;
    tcModel.vendorConsents.set(755, allowGoogle);
    return TCString.encode(tcModel);
  }

  /**
   * Create the consent banner element.
   */
  function createBanner(localeStrings) {
    const banner = document.createElement('div');
    banner.id = 'pcz-banner';
    banner.setAttribute('role', 'region');
    banner.setAttribute('aria-live', 'polite');
    banner.setAttribute('aria-label', localeStrings.bannerTitle);
    banner.innerHTML =
      '<div class="pcz-card">' +
        '<div class="pcz-copy">' +
          '<p class="pcz-body">' +
            localeStrings.bannerDesc +
            ' <button id="pcz-manage" type="button" class="pcz-link pcz-inline-link">' + localeStrings.managePrefs + '</button>' +
          '</p>' +
        '</div>' +
        '<div class="pcz-actions">' +
          '<button id="pcz-accept" type="button" class="pcz-btn pcz-primary">' + localeStrings.acceptAll + '</button>' +
          '<button id="pcz-reject" type="button" class="pcz-btn pcz-secondary">' + localeStrings.rejectAll + '</button>' +
        '</div>' +
        '<button id="pcz-close" type="button" class="pcz-close" aria-label="' + localeStrings.close + '"><span aria-hidden="true">&times;</span></button>' +
      '</div>';
    return banner;
  }

  /**
   * Create the preferences modal overlay.
   */
  function createModal(localeStrings, initialState) {
    const overlay = document.createElement('div');
    overlay.id = 'pcz-modal';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', localeStrings.modalTitle);
    const panel = document.createElement('div');
    panel.className = 'pcz-panel policy-card';
    panel.setAttribute('tabindex', '-1');
    let categoriesHTML = '';
    const stateOnLabel = localeStrings.stateOn || 'On';
    const stateOffLabel = localeStrings.stateOff || 'Off';
    const stateAlwaysOnLabel = localeStrings.stateAlwaysOn || 'Always on';
    const requiredLabel = localeStrings.requiredLabel || 'Required for site operation';
    ['necessary','analytics','functional','advertising'].forEach(function (key) {
      const cat = localeStrings.categories[key];
      const descId = 'pcz-pref-desc-' + key;
      const isDisabled = key === 'necessary';
      const checked = initialState[key] || isDisabled;
      const isOn = checked ? 'true' : 'false';
      const stateLabel = checked ? stateOnLabel : stateOffLabel;
      categoriesHTML +=
        '<div class="pref-option' + (isDisabled ? ' pref-option-locked' : '') + '" data-pref="' + key + '" role="listitem">' +
          '<div class="pref-option-head">' +
            (isDisabled
              ? '<div class="pref-status-row" data-locked="true" aria-disabled="true">' +
                  '<span class="pref-label-group">' +
                    '<span class="pref-label">' + cat.label + '</span>' +
                    '<span class="pref-helper">' + requiredLabel + '</span>' +
                  '</span>' +
                  '<span class="pref-state" aria-hidden="true">' + stateAlwaysOnLabel + '</span>' +
                '</div>'
              : '<button type="button" class="pref-toggle" data-pref="' + key + '" aria-pressed="' + isOn + '">' +
                  '<span class="pref-label-group">' +
                    '<span class="pref-label">' + cat.label + '</span>' +
                  '</span>' +
                  '<span class="pref-state" aria-hidden="true">' + stateLabel + '</span>' +
                '</button>') +
            '<button type="button" class="pref-info" aria-expanded="false" aria-controls="' + descId + '">' +
              '<span aria-hidden="true">?</span>' +
              '<span class="visually-hidden">Read what ' + cat.label + ' cookies do</span>' +
            '</button>' +
          '</div>' +
          '<div id="' + descId + '" class="pref-disclosure" aria-hidden="true">' +
            '<div class="pref-disclosure-inner">' +
              '<p class="pref-description">' + cat.description + '</p>' +
            '</div>' +
          '</div>' +
        '</div>';
    });
    const lead = localeStrings.modalLead ? '<p class="policy-lead">' + localeStrings.modalLead + '</p>' : '';
    const gpcNotice = hasGPC() ? '<p class="policy-lead pcz-gpc-notice">' + localeStrings.gpcHonoured + '</p>' : '';
    panel.innerHTML =
      '<div class="pcz-panel-head">' +
        '<h2>' + localeStrings.modalTitle + '</h2>' +
        '<button type="button" class="pcz-panel-close" id="pcz-close-modal" aria-label="' + localeStrings.closePrefs + '">' +
          '<span aria-hidden="true">&times;</span>' +
        '</button>' +
      '</div>' +
      lead +
      gpcNotice +
      '<div class="pref-grid" role="list">' + categoriesHTML + '</div>' +
      '<button type="button" id="pcz-save" class="pcz-save-preferences">' + localeStrings.savePrefs + '</button>';
    overlay.appendChild(panel);
    return overlay;
  }

  /**
   * Show the consent banner.
   */
  function showBanner(localeStrings) {
    const existingBanner = document.getElementById('pcz-banner');
    if (existingBanner && existingBanner.dataset.state !== 'closing') {
      setBannerUiState(true);
      return;
    }
    if (existingBanner) existingBanner.remove();
    const saved = loadConsent();
    const initialState = saved ? saved.categories : getDefaultState();
    const banner = createBanner(localeStrings);
    document.body.appendChild(banner);
    setBannerUiState(true);
    setTimeout(() => banner.classList.add('pcz-visible'), 16);
    // Banner is non-blocking; ensure the page isn't stuck in a blocked state.
    try { document.body.classList.remove('consent-blocked'); } catch (err) {}

    const closeBtn = banner.querySelector('#pcz-close');
    const acceptBtn = banner.querySelector('#pcz-accept');
    const rejectBtn = banner.querySelector('#pcz-reject');
    const manageBtn = banner.querySelector('#pcz-manage');

    const dismissBanner = () => {
      if (!banner || banner.dataset.state === 'closing') return;
      banner.dataset.state = 'closing';
      setBannerUiState(false);
      banner.classList.remove('pcz-visible');
      banner.classList.add('pcz-exit');
      const cleanup = () => {
        banner.remove();
      };
      banner.addEventListener('transitionend', cleanup, { once: true });
      banner.addEventListener('animationend', cleanup, { once: true });
      setTimeout(cleanup, 450);
      try { document.body.classList.remove('consent-blocked'); } catch (err) {}
    };

    function acceptAll() {
      const newState = {
        necessary: true,
        analytics: true,
        functional: true,
        advertising: hasGPC() ? false : true
      };
      saveConsent(newState);
      applyConsent(newState);
      dismissBanner();
    }

    function useEssentialOnly() {
      const newState = { necessary: true, analytics: false, functional: false, advertising: false };
      saveConsent(newState);
      applyConsent(newState);
      dismissBanner();
    }

    acceptBtn.addEventListener('click', acceptAll);
    if (closeBtn) {
      closeBtn.addEventListener('click', useEssentialOnly);
    }
    rejectBtn.addEventListener('click', useEssentialOnly);
    manageBtn.addEventListener('click', function (event) {
      event.preventDefault();
      dismissBanner();
      openPreferences(localeStrings, initialState, true);
    });
  }

  /**
   * Show the preferences modal and handle focus and save logic.
   */
  function openPreferences(localeStrings, currentState, blocking) {
    if (document.getElementById('pcz-modal')) return;
    if (blocking) {
      try { document.body.classList.add('consent-blocked'); } catch (err) {}
    }
    const modal = createModal(localeStrings, currentState);
    document.body.appendChild(modal);
    setTimeout(() => modal.classList.add('pcz-visible'), 16);
    const stateOnLabel = localeStrings.stateOn || 'On';
    const stateOffLabel = localeStrings.stateOff || 'Off';
    const focusable = modal.querySelectorAll('input, button');
    const firstFocusable = focusable[0];
    const lastFocusable = focusable[focusable.length - 1];
    modal.addEventListener('click', function (event) {
      if (event.target !== modal) return;
      closePreferences(modal);
      if (blocking) {
        showBanner(localeStrings);
      }
    });
    modal.addEventListener('keydown', function (e) {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstFocusable) {
            e.preventDefault();
            lastFocusable.focus();
          }
        } else {
          if (document.activeElement === lastFocusable) {
            e.preventDefault();
            firstFocusable.focus();
          }
        }
      } else if (e.key === 'Escape') {
        closePreferences(modal);
        if (blocking) {
          showBanner(localeStrings);
        }
      }
    });
    modal.querySelector('#pcz-close-modal').addEventListener('click', function () {
      closePreferences(modal);
      if (blocking) {
        showBanner(localeStrings);
      }
    });

    modal.querySelectorAll('.pref-toggle[data-pref]').forEach(function (button) {
      button.addEventListener('click', function () {
        if (button.disabled || button.dataset.locked === 'true') return;
        const current = button.getAttribute('aria-pressed') === 'true';
        button.setAttribute('aria-pressed', current ? 'false' : 'true');
        const stateEl = button.querySelector('.pref-state');
        if (stateEl) stateEl.textContent = current ? stateOffLabel : stateOnLabel;
      });
    });
    modal.querySelectorAll('.pref-info[aria-controls]').forEach(function (button) {
      button.addEventListener('click', function () {
        const expanded = button.getAttribute('aria-expanded') === 'true';
        const targetId = button.getAttribute('aria-controls');
        const target = targetId ? document.getElementById(targetId) : null;
        const row = button.closest('.pref-option');
        const nextExpanded = !expanded;
        button.setAttribute('aria-expanded', nextExpanded ? 'true' : 'false');
        if (row) row.classList.toggle('is-expanded', nextExpanded);
        if (target) target.setAttribute('aria-hidden', nextExpanded ? 'false' : 'true');
      });
    });

    modal.querySelector('#pcz-save').addEventListener('click', function () {
      const newState = {
        necessary: true,
        analytics: false,
        functional: false,
        advertising: false
      };
      modal.querySelectorAll('.pref-toggle[data-pref]').forEach(function (button) {
        const key = String(button.dataset.pref || '').trim();
        if (!key || key === 'necessary') return;
        newState[key] = button.getAttribute('aria-pressed') === 'true';
      });
      if (hasGPC()) newState.advertising = false;
      saveConsent(newState);
      applyConsent(newState);
      closePreferences(modal);
      document.body.classList.remove('consent-blocked');
    });
    firstFocusable.focus();
  }

  /**
   * Remove the preferences modal from the DOM.
   */
  function closePreferences(modal) {
    if (!modal || modal.dataset.state === 'closing') return;
    modal.dataset.state = 'closing';
    modal.classList.remove('pcz-visible');
    modal.classList.add('pcz-exit');
    const cleanup = () => {
      try { modal.remove(); } catch (err) {}
    };
    modal.addEventListener('transitionend', cleanup, { once: true });
    modal.addEventListener('animationend', cleanup, { once: true });
    setTimeout(cleanup, 450);
  }

  /**
   * Initialize the CMP. Apply saved consent or show banner.
   */
  function init() {
    if (isEmbeddedSameOrigin()) {
      setBannerUiState(false);
      const saved = loadConsent();
      const embeddedState = Object.assign(
        {},
        saved && saved.categories ? saved.categories : getDefaultState(),
        { analytics: false, advertising: false }
      );
      applyConsent(embeddedState);
      return;
    }
    loadStyles();
    const locale = getLocale();
    const localeStrings = CONFIG.languages[locale];
    if (GLOBAL_CONF.ui && GLOBAL_CONF.ui.persistLinkSelector) {
      const persistLinks = document.querySelectorAll(GLOBAL_CONF.ui.persistLinkSelector);
      persistLinks.forEach(function (link) {
        if (!link || link.dataset.prefNavBound === 'true') return;
        link.dataset.prefNavBound = 'true';
        link.addEventListener('click', function (event) {
          event.preventDefault();
          const saved = loadConsent();
          const currentState = saved && saved.categories ? saved.categories : getDefaultState();
          openPreferences(localeStrings, currentState, false);
        });
      });
    }
    // Dev/testing aids via URL parameters
    try {
      const q = new URLSearchParams(location.search);
      if (q.get('reset_consent') === '1') {
        localStorage.removeItem(STORAGE_KEY);
      }
      if (q.get('show_consent') === '1') {
        showBanner(localeStrings);
        return;
      }
    } catch (e) {}
    const saved = loadConsent();
    if (saved) {
      setBannerUiState(false);
      if (hasGPC() && saved.categories && saved.categories.advertising) {
        saved.categories.advertising = false;
        saveConsent(saved.categories);
      }
      applyConsent(saved.categories);
    } else {
      setBannerUiState(false);
      const defaultState = getDefaultState();
      applyConsent(defaultState);
      showBanner(localeStrings);
    }
  }

  // Expose a simple public API
  window.consentAPI = {
    open: function () {
      loadStyles();
      const locale = getLocale();
      const localeStrings = CONFIG.languages[locale];
      const saved = loadConsent();
      const currentState = saved && saved.categories ? saved.categories : getDefaultState();
      openPreferences(localeStrings, currentState, false);
    },
    get: function () {
      const saved = loadConsent();
      return saved ? saved.categories : null;
    },
    set: function (prefs) {
      const state = Object.assign({ necessary: true, analytics: false, functional: false, advertising: false }, prefs);
      saveConsent(state);
      applyConsent(state);
    },
    reset: function () {
      localStorage.removeItem(STORAGE_KEY);
      init();
    }
  };

  // Initialise when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
