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

  const STYLE_ID = 'pcz-consent-styles';

  function loadStyles() {
    if (document.getElementById(STYLE_ID) || document.querySelector('link[href$="privacy.css"]')) return;
    const link = document.createElement('link');
    link.id = STYLE_ID;
    link.rel = 'stylesheet';
    link.href = 'css/privacy.css';
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
        bannerTitle: 'I value your privacy',
        bannerDesc: "I use cookies and similar technologies to improve your experience, understand site traffic, and measure performance. You can allow all cookies, allow essential cookies only, or manage your settings.",
        acceptAll: 'Allow all cookies',
        rejectAll: 'Allow essential only',
        managePrefs: 'Manage settings',
        privacyPolicy: 'Privacy Policy',
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
        bannerDesc: 'Utilizo cookies y tecnologías similares para mejorar tu experiencia, entender el tráfico del sitio y medir el rendimiento. Puedes permitir todas las cookies, permitir solo las esenciales o administrar tus ajustes.',
        acceptAll: 'Permitir todas las cookies',
        rejectAll: 'Permitir solo las esenciales',
        managePrefs: 'Administrar ajustes',
        privacyPolicy: 'Política de privacidad',
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
  function getRegion() {
    return 'US';
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
    window.dispatchEvent(new CustomEvent('consent-changed', { detail: state }));
  }

  /**
   * Dynamically load or unload a vendor script based on consent.
   */
  function enableVendor(vendorKey, enabled) {
    const vendor = CONFIG.vendors[vendorKey];
    if (!vendor) return;
    if (enabled && !vendor.enabled) {
      if (vendorKey === 'ga4') {
        const scriptId = 'ga4-src';
        if (!document.getElementById(scriptId)) {
          const s = document.createElement('script');
          s.id = scriptId;
          s.async = true;
          s.src = 'https://www.googletagmanager.com/gtag/js?id=' + vendor.id;
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
    const initialState = saved ? saved.categories : { necessary: true, analytics: false, functional: false, advertising: false };
    const banner = createBanner(localeStrings);
    document.body.appendChild(banner);
    // Block page interaction until a choice is made
    document.body.classList.add('consent-blocked');
    const acceptBtn = banner.querySelector('#pcz-accept');
    const rejectBtn = banner.querySelector('#pcz-reject');
    const manageBtn = banner.querySelector('#pcz-manage');
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
    acceptBtn.addEventListener('click', function () {
      const newState = { necessary: true, analytics: true, functional: true, advertising: true };
      saveConsent(newState);
      applyConsent(newState);
      banner.remove();
      document.body.classList.remove('consent-blocked');
    });
    rejectBtn.addEventListener('click', function () {
      const newState = { necessary: true, analytics: false, functional: false, advertising: false };
      // If GPC is enabled, advertising must remain false
      if (hasGPC()) newState.advertising = false;
      saveConsent(newState);
      applyConsent(newState);
      banner.remove();
      document.body.classList.remove('consent-blocked');
    });
    manageBtn.addEventListener('click', function () {
      openPreferences(localeStrings, initialState, true);
      banner.remove();
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
    const saved = loadConsent();
    if (hasGPC()) {
      if (saved && saved.categories && saved.categories.advertising) {
        saved.categories.advertising = false;
        saveConsent(saved.categories);
        applyConsent(saved.categories);
      }
    }
    if (saved) {
      applyConsent(saved.categories);
    } else {
      showBanner(localeStrings);
    }
  }

  // Expose a simple public API
  window.consentAPI = {
    open: function () {
      loadStyles();
      const locale = getLocale();
      const localeStrings = CONFIG.languages[locale];
      const saved = loadConsent() || { categories: { necessary: true, analytics: false, functional: false, advertising: false } };
      openPreferences(localeStrings, saved.categories);
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
