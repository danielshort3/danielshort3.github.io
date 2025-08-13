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

  /**
   * Configuration for the CMP. You can extend this object with
   * additional languages or categories as needed. Titles and
   * descriptions should be plain language with no dark patterns.
   */
  const CONFIG = {
    version: 1,
    languages: {
      en: {
        bannerTitle: 'We value your privacy',
        bannerDesc: 'We use cookies and similar technologies to enhance your browsing experience, analyse site traffic and measure campaign performance. You can accept all, reject all, or customise your preferences.',
        acceptAll: 'Accept all',
        rejectAll: 'Reject all',
        managePrefs: 'Manage preferences',
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
        bannerTitle: 'Valoramos su privacidad',
        bannerDesc: 'Usamos cookies y tecnologías similares para mejorar su experiencia de navegación, analizar el tráfico del sitio y medir el rendimiento de campañas. Puede aceptar todo, rechazar todo o personalizar sus preferencias.',
        acceptAll: 'Aceptar todo',
        rejectAll: 'Rechazar todo',
        managePrefs: 'Administrar preferencias',
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
        localStorage.setItem('consent', JSON.stringify(record));
      });
      return;
    }
    localStorage.setItem('consent', JSON.stringify(record));
  }

  /**
   * Load a previously saved consent record from localStorage.
   */
  function loadConsent() {
    try {
      const raw = localStorage.getItem('consent');
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
    banner.id = 'privacy-banner';
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-label', localeStrings.bannerTitle);
    Object.assign(banner.style, {
      position: 'fixed',
      bottom: '0',
      left: '0',
      right: '0',
      backgroundColor: '#f7f7f7',
      borderTop: '1px solid #ccc',
      padding: '1rem',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'column',
      gap: '0.5rem',
      boxShadow: '0 -2px 5px rgba(0,0,0,0.1)'
    });
    banner.innerHTML = '' +
      '<strong style="font-size:1rem;">' + localeStrings.bannerTitle + '</strong>' +
      '<p style="margin:0;font-size:0.875rem;line-height:1.4;">' + localeStrings.bannerDesc + '</p>' +
      '<div style="margin-top:0.5rem;display:flex;gap:0.5rem;flex-wrap:wrap;">' +
        '<button id="privacy-accept" style="padding:0.5rem 1rem;border:none;border-radius:4px;background:#0a84ff;color:white;cursor:pointer;">' + localeStrings.acceptAll + '</button>' +
        '<button id="privacy-reject" style="padding:0.5rem 1rem;border:none;border-radius:4px;background:#ccc;color:#333;cursor:pointer;">' + localeStrings.rejectAll + '</button>' +
        '<button id="privacy-manage" style="padding:0.5rem 1rem;border:none;border-radius:4px;background:transparent;color:#0a84ff;text-decoration:underline;cursor:pointer;">' + localeStrings.managePrefs + '</button>' +
      '</div>';
    return banner;
  }

  /**
   * Create the preferences modal overlay.
   */
  function createModal(localeStrings, initialState) {
    const overlay = document.createElement('div');
    overlay.id = 'privacy-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', localeStrings.modalTitle);
    Object.assign(overlay.style, {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)',
      zIndex: 1001,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    });
    const modal = document.createElement('div');
    Object.assign(modal.style, {
      backgroundColor: 'white',
      borderRadius: '6px',
      maxWidth: '480px',
      width: '90%',
      padding: '1rem',
      maxHeight: '90%',
      overflowY: 'auto'
    });
    let categoriesHTML = '';
    ['necessary','analytics','functional','advertising'].forEach(function (key) {
      const cat = localeStrings.categories[key];
      const isDisabled = key === 'necessary';
      const checked = initialState[key] || isDisabled;
      categoriesHTML += '' +
        '<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:0.75rem;">' +
          '<div style="max-width:80%;">' +
            '<strong style="font-size:0.9rem;">' + cat.label + '</strong>' +
            '<p style="margin:0;font-size:0.8rem;line-height:1.4;color:#555;">' + cat.description + '</p>' +
          '</div>' +
          '<div>' +
            '<input type="checkbox" id="toggle-' + key + '" name="' + key + '" ' + (checked ? 'checked' : '') + ' ' + (isDisabled ? 'disabled aria-disabled="true"' : '') + '/>' +
          '</div>' +
        '</div>';
    });
    const gpcNotice = hasGPC() ? '<p style="background:#ffeebe;padding:0.5rem;border-radius:4px;font-size:0.75rem;margin-bottom:0.75rem;">' + localeStrings.gpcHonoured + '</p>' : '';
    modal.innerHTML = '' +
      '<h2 style="margin-top:0;">' + localeStrings.modalTitle + '</h2>' +
      gpcNotice +
      '<form id="privacy-form">' + categoriesHTML + '</form>' +
      '<div style="display:flex;justify-content:flex-end;gap:0.5rem;margin-top:1rem;">' +
        '<button type="button" id="privacy-cancel" style="padding:0.5rem 1rem;border:none;border-radius:4px;background:#ccc;color:#333;cursor:pointer;">' + localeStrings.cancel + '</button>' +
        '<button type="button" id="privacy-save" style="padding:0.5rem 1rem;border:none;border-radius:4px;background:#0a84ff;color:white;cursor:pointer;">' + localeStrings.savePrefs + '</button>' +
      '</div>';
    overlay.appendChild(modal);
    return overlay;
  }

  /**
   * Show the consent banner.
   */
  function showBanner(localeStrings) {
    if (document.getElementById('privacy-banner')) return;
    const saved = loadConsent();
    const initialState = saved ? saved.categories : { necessary: true, analytics: false, functional: false, advertising: false };
    const banner = createBanner(localeStrings);
    document.body.appendChild(banner);
    const acceptBtn = banner.querySelector('#privacy-accept');
    const rejectBtn = banner.querySelector('#privacy-reject');
    const manageBtn = banner.querySelector('#privacy-manage');
    acceptBtn.addEventListener('click', function () {
      const newState = { necessary: true, analytics: true, functional: true, advertising: true };
      saveConsent(newState);
      applyConsent(newState);
      banner.remove();
    });
    rejectBtn.addEventListener('click', function () {
      const newState = { necessary: true, analytics: false, functional: false, advertising: false };
      // If GPC is enabled, advertising must remain false
      if (hasGPC()) newState.advertising = false;
      saveConsent(newState);
      applyConsent(newState);
      banner.remove();
    });
    manageBtn.addEventListener('click', function () {
      openPreferences(localeStrings, initialState);
      banner.remove();
    });
  }

  /**
   * Show the preferences modal and handle focus and save logic.
   */
  function openPreferences(localeStrings, currentState) {
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
      }
    });
    modal.querySelector('#privacy-cancel').addEventListener('click', function () {
      closePreferences(modal);
    });
    modal.querySelector('#privacy-save').addEventListener('click', function () {
      const form = modal.querySelector('#privacy-form');
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
      localStorage.removeItem('consent');
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