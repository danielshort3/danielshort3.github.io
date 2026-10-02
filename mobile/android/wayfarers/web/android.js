(function () {
  'use strict';
  function viewport() {
    document.documentElement.style.setProperty('--android-webview-height', window.innerHeight + 'px');
  }
  viewport();
  window.addEventListener('resize', viewport);
  function send(type, text) {
    if (window.WayfarersAndroid) window.WayfarersAndroid.postMessage(JSON.stringify({ type: type, text: text || '' }));
  }
  const options = document.querySelector('.wg-exit');
  options.href = '#app-options';
  options.setAttribute('aria-label', 'App options and updates');
  options.innerHTML = '<span aria-hidden="true" style="display:block">⋮</span>';
  options.addEventListener('click', function (event) { event.preventDefault(); send('options'); });
  document.addEventListener('click', function (event) {
    const button = event.target.closest('[data-export="download"]');
    if (!button) return;
    // The canonical handler serializes and validates the guild before native export.
    const input = document.querySelector('#wg-save-text');
    const previous = input.value;
    input.value = '';
    button.dataset.export = 'text';
    setTimeout(function () {
      button.dataset.export = 'download';
      const text = input.value;
      if (text) send('export', text);
      else input.value = previous;
    }, 0);
  }, true);
  const observer = new MutationObserver(function () {
    const body = document.querySelector('[data-dialog-body]');
    if (!body || !body.querySelector('#wg-save-text') || body.querySelector('[data-native-options]')) return;
    const actions = document.createElement('div');
    actions.className = 'wg-dialog-actions';
    actions.dataset.nativeOptions = '';
    for (const item of [['options', 'App updates'], ['import', 'Open save file']]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'wg-button';
      button.textContent = item[1];
      button.addEventListener('click', function () { send(item[0]); });
      actions.append(button);
    }
    body.prepend(actions);
    const file = body.querySelector('input[type="file"]');
    if (file) {
      file.hidden = true;
      const label = body.querySelector('label[for="' + file.id + '"]');
      if (label) label.hidden = true;
    }
  });
  observer.observe(document.querySelector('[data-dialog-body]'), { childList: true, subtree: true });
  window.WayfarersAndroidUI = {
    reviewImport: function (text) {
      document.querySelector('[data-open="settings"]').click();
      const input = document.querySelector('#wg-save-text');
      input.value = text;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      document.querySelector('[data-review-import]').click();
    },
    flush: function () {
      document.dispatchEvent(new Event('visibilitychange'));
      const warning = document.querySelector('[data-storage-warning]');
      const stored = window.localStorage.getItem(window.WayfarersStorage.SAVE_KEY);
      return Boolean(stored && (!warning || warning.hidden));
    }
  };
})();
