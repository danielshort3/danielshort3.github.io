/*
 * consent_manager.js
 *
 * A lightweight, privacy‑focused consent management implementation for
 * static websites. It displays an accessible banner and preferences
 * modal, persists the user’s choices, integrates with Google Consent
 * Mode v2 and the IAB TCF v2.2 (when appropriate), and honors
 * Global Privacy Control (GPC) signals for California residents.
 *
 * This script does not require any external dependencies aside from
 * Google’s gtag.js (which should be loaded with default denied
 * settings) and the optional @iabtcf/core library if you choose to
 * generate a TCF string. See inline comments for configuration
 * guidance.
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
  const PREFERENCES_URL = 'privacy.html#prefs-title';
  const currentScript = document.currentScript || document.querySelector('script[src*="consent_manager"]');
  const scriptSrc = currentScript ? currentScript.getAttribute('src') || '' : '';
  const assetVariant = /\/dist\/js\//.test(scriptSrc) ? 'dist/js/' : 'js/';
  const resolveAsset = (src) => {
    if (!src) return src;
    if (/^(?:https?:)?\/\//.test(src)) return src;
    if (src.startsWith('/')) return src;
    return assetVariant === 'dist/js/' ? src.replace(/^js\//, 'dist/js/') : src;
  };

  function loadStyles() {
    if (document.getElementById(STYLE_ID) || document.querySelector('link[href$="privacy.css"]')) return;
    const link = document.createElement('link');
    link.id = STYLE_ID;
    link.rel = 'stylesheet';
    link.href = 'css/privacy.css';
    document.head.appendChild(link);
  }

  function goToPreferencePage() {
    try {
      document.body.classList.remove('consent-blocked');
    } catch (err) {}
    try {
      window.location.assign(PREFERENCES_URL);
    } catch (err) {
      window.location.href = PREFERENCES_URL;
    }
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
        bannerDesc: "I use cookies and similar technologies to improve your experience, understand site traffic, and measure performance.",
        acceptAll: 'Allow all cookies',
        rejectAll: 'Allow essential only',
        managePrefs: 'Manage settings',
        privacyPolicy: 'Privacy Policy',
        close: 'Close banner and accept all cookies',
        modalTitle: 'Privacy preferences',
        savePrefs: 'Save preferences',
        cancel: 'Cancel',
        categories: {
          necessary: {
            label: 'Strictly necessary',
            description: 'These cookies are required for basic website functions such as navigation and access to secure areas. You cannot turn them off.'
          },
          analytics: {
            label: 'Analytics',
            description: 'Analytics cookies help us understand how visitors interact with our website and improve performance.'
          },
          functional: {
            label: 'Functional',
            description: 'Functional cookies allow the website to remember your choices and provide enhanced, more personal features.'
          },
          advertising: {
            label: 'Advertising',
            description: 'Advertising cookies are used to deliver relevant ads and to measure the effectiveness of marketing campaigns.'
          }
        },
        doNotSell: 'Do Not Sell/Share My Personal Information',
        gpcHonoured: 'Your browser sent a Global Privacy Control signal so we\'ve applied your opt‑out. You can adjust other preferences below.'
      },
      es: {
        bannerTitle: 'Valoro tu privacidad',
        bannerDesc: 'Utilizo cookies y tecnologías similares para mejorar tu experiencia, entender el tráfico del sitio y medir el rendimiento. Al cerrar este aviso o elegir Permitir todas aceptas las cookies; puedes cambiar tu decisión en Administrar ajustes.',
        acceptAll: 'Permitir todas las cookies',
        rejectAll: 'Permitir solo las esenciales',
        managePrefs: 'Administrar ajustes',
        privacyPolicy: 'Política de privacidad',
        close: 'Cerrar y aceptar todas las cookies',
        modalTitle: 'Preferencias de privacidad',
        savePrefs: 'Guardar preferencias',
        cancel: 'Cancelar',
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
      ga4: {
        id: 'G-0VL37MQ62P',
        enabled: false
      }
    }
  };

  const GLOBAL_CONF = window.PrivacyConfig || {};
  if (GLOBAL_CONF.vendors) {
    Object.keys(GLOBAL_CONF.vendors).forEach(function (v) {
      CONFIG.vendors[v] = Object.assign(CONFIG.vendors[v] || {}, GLOBAL_CONF.vendors[v]);
    });
  }
  const STORAGE_KEY = GLOBAL_CONF.storageKey || 'consent';

  const loadAnalyticsHelpers = (() => {
    let promise = null;
    return function loadHelpers(){
      if (promise) return promise;
      promise = new Promise((resolve, reject) => {
        if (document.getElementById('ga4-helper')) {
          resolve();
          return;
        }
        const tag = document.createElement('script');
        tag.id = 'ga4-helper';
        tag.src = resolveAsset('js/analytics/ga4-events.js');
        tag.async = false;
        tag.onload = () => resolve();
        tag.onerror = () => {
          promise = null;
          reject(new Error('Failed to load ga4-events.js'));
        };
        document.head.appendChild(tag);
      });
      return promise;
    };
  })();

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

  function getDefaultState() {
    const strictState = { necessary: true, analytics: false, functional: false, advertising: false };
    const permissiveState = { necessary: true, analytics: true, functional: true, advertising: true };
    const region = getRegion();
    const stateCode = getUSState();
    const base = regionRequiresOptIn(region, stateCode) ? strictState : permissiveState;
    const result = Object.assign({}, base);
    if (hasGPC()) {
      result.advertising = false;
    }
    return result;
  }

  /**
   * Persist the consent record in localStorage. The record includes
   * timestamp, region, version, user categories, GPC status and a
   * placeholder for the TCF string when applicable.
   */
  function saveConsent(state) {
    const record = {
      version: CONFIG.version,
      timestamp: Date.now(),
      region: getRegion(),
      gpc: hasGPC(),
      categories: state,
      tcString: ''
    };
    if (record.region === 'EU' && window.__iabtcf) {
      getTcString(state).then(tc => {
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
      return raw ? JSON.parse(raw) : null;
    } catch (err) {
      return null;
    }
  }

  /**
   * Apply the chosen consent state by updating Google Consent Mode
   * and enabling or disabling vendors.
   */
  function applyConsent(state) {
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
    enableVendor('ga4', state.analytics);
    enableVendor('hotjar', state.analytics);
    window.dispatchEvent(new CustomEvent('consent-changed', { detail: state }));
  }

  /**
   * Dynamically load or unload a vendor script based on consent.
   */
  function enableVendor(vendorKey, enabled) {
    const vendor = CONFIG.vendors[vendorKey];
    if (!vendor) return;
    if (enabled) {
      loadAnalyticsHelpers()
        .then(() => {
          if (window.consentAPI && typeof window.consentAPI.get === 'function') {
            const current = window.consentAPI.get();
            if (current) {
              window.dispatchEvent(new CustomEvent('consent-changed', { detail: current }));
            }
          }
        })
        .catch(err => console.warn(err));
    }
    if (enabled && !vendor.enabled) {
      if (vendorKey === 'ga4') {
        const scriptId = 'ga4-src';
        if (!document.getElementById(scriptId)) {
          const s = document.createElement('script');
          s.id = scriptId;
          s.async = true;
          s.src = 'https://www.googletagmanager.com/gtag/js?id=' + vendor.id;
          s.onload = function(){
            // Ensure global gtag shim exists, then configure the property
            window.dataLayer = window.dataLayer || [];
            window.gtag = window.gtag || function(){ (window.dataLayer = window.dataLayer || []).push(arguments); };
            try {
              window.gtag('js', new Date());
              window.gtag('config', vendor.id);
            } catch {}
          };
          document.head.appendChild(s);
        }
      } else if (vendorKey === 'hotjar') {
        if (!vendor.id) return;
        document.body.dataset.hotjarSite = vendor.id;
        const tagId = 'hotjar-loader';
        if (!document.getElementById(tagId)) {
          const s = document.createElement('script');
          s.id = tagId;
          s.defer = true;
          s.src = resolveAsset('js/analytics/hotjar.js');
          document.head.appendChild(s);
        }
      }
      vendor.enabled = true;
    }
    if (!enabled && vendor.enabled) {
      // GA4 cannot be fully unloaded; rely on consent update.
      vendor.enabled = false;
    }
  }

  /**
   * Generate a TCF v2.2 compliant string using the @iabtcf/core library.
   */
  async function getTcString(state) {
    const api = window.__iabtcf;
    if (!api) return '';
    const { TCModel, TCString, GVL } = api;
    const gvl = new GVL();
    await gvl.readyPromise;
    const tcModel = new TCModel(gvl);
    tcModel.cmpId = 123;
    tcModel.cmpVersion = 1;
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
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-modal', 'true');
    banner.setAttribute('aria-label', localeStrings.bannerTitle);
    banner.innerHTML =
      '<button id="pcz-close" type="button" class="pcz-close" aria-label="' + localeStrings.close + '"><span aria-hidden="true">&times;</span></button>' +
      '<div class="pcz-row">' +
        '<p><strong>' + localeStrings.bannerTitle + '</strong> ' + localeStrings.bannerDesc + ' <a class="pcz-link" href="privacy.html">' + localeStrings.privacyPolicy + '</a></p>' +
        '<button id="pcz-accept" class="pcz-btn pcz-primary">' + localeStrings.acceptAll + '</button>' +
        '<button id="pcz-reject" class="pcz-btn pcz-secondary">' + localeStrings.rejectAll + '</button>' +
        '<button id="pcz-manage" class="pcz-btn pcz-link">' + localeStrings.managePrefs + '</button>' +
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
    panel.className = 'pcz-panel';
    panel.setAttribute('tabindex', '-1');
    let categoriesHTML = '';
    ['necessary','analytics','functional','advertising'].forEach(function (key) {
      const cat = localeStrings.categories[key];
      const isDisabled = key === 'necessary';
      const checked = initialState[key] || isDisabled;
      categoriesHTML +=
        '<fieldset>' +
          '<legend>' + cat.label + '</legend>' +
          '<label>' +
            '<input type="checkbox" id="toggle-' + key + '" name="' + key + '" ' + (checked ? 'checked' : '') + ' ' + (isDisabled ? 'disabled aria-disabled="true"' : '') + '/>' +
            '<span class="pcz-helper">' + cat.description + '</span>' +
          '</label>' +
        '</fieldset>';
    });
    const gpcNotice = hasGPC() ? '<p class="pcz-helper">' + localeStrings.gpcHonoured + '</p>' : '';
    panel.innerHTML =
      '<h2>' + localeStrings.modalTitle + '</h2>' +
      gpcNotice +
      '<form id="pcz-form">' + categoriesHTML + '</form>' +
      '<div class="pcz-actions">' +
        '<button type="button" id="pcz-cancel" class="pcz-btn pcz-secondary">' + localeStrings.cancel + '</button>' +
        '<button type="button" id="pcz-save" class="pcz-btn pcz-primary">' + localeStrings.savePrefs + '</button>' +
      '</div>';
    overlay.appendChild(panel);
    return overlay;
  }

  /**
   * Show the consent banner.
   */
  function showBanner(localeStrings) {
    if (document.getElementById('pcz-banner')) return;
    const saved = loadConsent();
    const initialState = saved ? saved.categories : getDefaultState();
    const banner = createBanner(localeStrings);
    document.body.appendChild(banner);
    // Block page interaction until a choice is made
    document.body.classList.add('consent-blocked');
    const closeBtn = banner.querySelector('#pcz-close');
    const acceptBtn = banner.querySelector('#pcz-accept');
    const rejectBtn = banner.querySelector('#pcz-reject');
    const manageBtn = banner.querySelector('#pcz-manage');
    const dismissBanner = () => {
      if (!banner || banner.dataset.state === 'closing') return;
      banner.dataset.state = 'closing';
      banner.classList.add('pcz-exit');
      const cleanup = () => {
        banner.remove();
      };
      banner.addEventListener('transitionend', cleanup, { once: true });
      banner.addEventListener('animationend', cleanup, { once: true });
      setTimeout(cleanup, 450);
      document.body.classList.remove('consent-blocked');
    };
    // Focus trap within the banner while blocking is active
    const focusables = banner.querySelectorAll('a,button,[tabindex]:not([tabindex="-1"])');
    const first = focusables[0];
    const last  = focusables[focusables.length - 1];
    banner.addEventListener('keydown', function(e){
      if (e.key !== 'Tab') return;
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
    // Set initial focus to the privacy‑preserving choice
    (rejectBtn || acceptBtn || manageBtn).focus();
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
    acceptBtn.addEventListener('click', acceptAll);
    if (closeBtn) {
      closeBtn.addEventListener('click', acceptAll);
    }
    rejectBtn.addEventListener('click', function () {
      const newState = { necessary: true, analytics: false, functional: false, advertising: false };
      // If GPC is enabled, advertising must remain false
      if (hasGPC()) newState.advertising = false;
      saveConsent(newState);
      applyConsent(newState);
      dismissBanner();
    });
    manageBtn.addEventListener('click', function (event) {
      event.preventDefault();
      dismissBanner();
      goToPreferencePage();
    });
  }

  /**
   * Show the preferences modal and handle focus and save logic.
   */
  function openPreferences(localeStrings, currentState, blocking) {
    const modal = createModal(localeStrings, currentState);
    document.body.appendChild(modal);
    const focusable = modal.querySelectorAll('input, button');
    const firstFocusable = focusable[0];
    const lastFocusable = focusable[focusable.length - 1];
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
    modal.querySelector('#pcz-cancel').addEventListener('click', function () {
      closePreferences(modal);
      if (blocking) {
        showBanner(localeStrings);
      }
    });
    modal.querySelector('#pcz-save').addEventListener('click', function () {
      const form = modal.querySelector('#pcz-form');
      const formData = new FormData(form);
      const newState = {
        necessary: true,
        analytics: formData.has('analytics'),
        functional: formData.has('functional'),
        advertising: formData.has('advertising')
      };
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
    modal.remove();
  }

  /**
   * Initialize the CMP. Apply saved consent or show banner.
   */
  function init() {
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
          goToPreferencePage();
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
      if (hasGPC() && saved.categories && saved.categories.advertising) {
        saved.categories.advertising = false;
        saveConsent(saved.categories);
      }
      applyConsent(saved.categories);
    } else {
      const defaultState = getDefaultState();
      applyConsent(defaultState);
      showBanner(localeStrings);
    }
  }

  // Expose a simple public API
  window.consentAPI = {
    open: function () {
      goToPreferencePage();
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
