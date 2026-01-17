(() => {
  'use strict';

  const TOGGLE_SELECTOR = '.pref-toggle[data-pref]';
  const INFO_SELECTOR = '.pref-info[aria-controls]';

  const updateToggleUI = (button, value) => {
    if (!button) return;
    const locked = button.dataset.locked === 'true';
    const pressed = locked ? true : Boolean(value);
    button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
    const stateEl = button.querySelector('.pref-state');
    if (stateEl) stateEl.textContent = pressed ? 'On' : 'Off';
  };

  const getCurrentPrefs = () => {
    const prefs = { analytics: false, functional: false, advertising: false };
    document.querySelectorAll(TOGGLE_SELECTOR).forEach((button) => {
      const pref = button.dataset.pref;
      if (!pref || pref === 'necessary') return;
      prefs[pref] = button.getAttribute('aria-pressed') === 'true';
    });
    return prefs;
  };

  const syncForm = () => {
    if (!window.consentAPI || typeof window.consentAPI.get !== 'function') {
      setTimeout(syncForm, 120);
      return;
    }
    const prefs = window.consentAPI.get() || {};
    document.querySelectorAll(TOGGLE_SELECTOR).forEach((button) => {
      const pref = button.dataset.pref;
      const value = pref === 'necessary' ? true : Boolean(prefs[pref]);
      updateToggleUI(button, value);
    });
  };

  const bindToggleEvents = () => {
    document.querySelectorAll(TOGGLE_SELECTOR).forEach((button) => {
      if (button.dataset.prefBound === 'true') return;
      button.dataset.prefBound = 'true';
      button.addEventListener('click', () => {
        if (button.disabled || button.dataset.locked === 'true') return;
        const current = button.getAttribute('aria-pressed') === 'true';
        updateToggleUI(button, !current);
      });
    });
  };

  const bindInfoButtons = () => {
    document.querySelectorAll(INFO_SELECTOR).forEach((button) => {
      if (button.dataset.prefInfoBound === 'true') return;
      button.dataset.prefInfoBound = 'true';
      button.addEventListener('click', () => {
        const expanded = button.getAttribute('aria-expanded') === 'true';
        const targetId = button.getAttribute('aria-controls');
        const target = targetId ? document.getElementById(targetId) : null;
        button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
        if (target) target.hidden = expanded;
      });
    });
  };

  const bindSaveButton = () => {
    const saveBtn = document.getElementById('save-privacy-preferences');
    if (!saveBtn || saveBtn.dataset.prefSaveBound === 'true') return;
    saveBtn.dataset.prefSaveBound = 'true';
    saveBtn.addEventListener('click', () => {
      if (!window.consentAPI || typeof window.consentAPI.set !== 'function') return;
      const newPrefs = getCurrentPrefs();
      window.consentAPI.set(newPrefs);
      const statusEl = document.getElementById('privacy-preferences-status');
      if (statusEl) statusEl.textContent = 'Your preferences have been saved.';
    });
  };

  document.addEventListener('DOMContentLoaded', () => {
    bindToggleEvents();
    bindInfoButtons();
    bindSaveButton();
    syncForm();
  });
})();

