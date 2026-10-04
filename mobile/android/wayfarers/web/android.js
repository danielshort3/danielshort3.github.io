(function () {
  'use strict';
  function viewport() {
    document.documentElement.style.setProperty('--android-webview-height', window.innerHeight + 'px');
  }
  viewport();
  window.addEventListener('resize', viewport);
  function send(type, text) {
    if (window.WayfarersAndroid) window.WayfarersAndroid.postMessage(JSON.stringify({ type: type, text: text || '',
      documentToken: window.WayfarersContent && window.WayfarersContent.documentToken }));
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
    prepareContentUpdate: function () {
      if (!window.WayfarersUI || !window.WayfarersUI.suspendForContentUpdate()) return null;
      const storage = {};
      for (let index = 0; index < localStorage.length; index++) {
        const key = localStorage.key(index);
        if (key && key.startsWith('wayfarers-guild-')) storage[key] = localStorage.getItem(key);
      }
      const text = window.WayfarersCheckpoint && window.WayfarersCheckpoint.snapshot();
      return text ? { text, storage, generation: window.WayfarersCheckpoint.generation() } : null;
    },
    durableSnapshot: function () {
      if (!window.WayfarersAndroidUI.flush()) return '';
      return window.WayfarersCheckpoint ? window.WayfarersCheckpoint.snapshot() : '';
    },
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
  // Readiness requires the canonical game and a durable native save.
  if (window.WayfarersContent && window.WayfarersAndroid) {
    function sceneReady() {
      const world = document.querySelector('[data-wx-station-world]');
      if (!world) {
        const canvas = document.querySelector('[data-wx-canvas]');
        return !canvas || canvas.dataset.sceneStatus === 'ready';
      }
      const clip = world.getBoundingClientRect();
      const visible = Array.from(world.querySelectorAll('canvas')).filter(function (canvas) {
        const rect = canvas.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && rect.right > Math.max(0, clip.left) &&
          rect.left < Math.min(window.innerWidth, clip.right) && rect.bottom > Math.max(0, clip.top) &&
          rect.top < Math.min(window.innerHeight, clip.bottom);
      });
      // Offscreen stations render lazily. The legacy compatibility canvas must
      // never certify a newly visible scene before its real art has painted.
      return visible.length > 0 && visible.every(function (canvas) { return canvas.dataset.sceneStatus === 'ready'; });
    }
    let flushed = false;
    const timer = setInterval(function () {
      if (window.WayfarersContent.restoreFailed || !window.WayfarersUI || !window.WayfarersUI.contentReady()) return;
      if (!sceneReady()) return;
      if (!flushed) { flushed = window.WayfarersUI.flushForContentUpdate(); return; }
      if (!window.WayfarersCheckpoint.confirmed()) return;
      clearInterval(timer);
      window.WayfarersAndroid.postMessage(JSON.stringify({ type: 'content-ready',
        documentToken: window.WayfarersContent.documentToken,
        version: window.WayfarersContent.version,
        recoveryToken: window.WayfarersContent.recoveryToken || '',
        text: window.WayfarersCheckpoint.snapshot(), generation: window.WayfarersCheckpoint.generation() }));
    }, 150);
    window.addEventListener('pagehide', function () { clearInterval(timer); }, { once: true });
  }
})();
