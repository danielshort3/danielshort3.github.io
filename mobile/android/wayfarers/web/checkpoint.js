(function () {
  'use strict';
  const api = window.WayfarersStorage;
  const createStore = api.createStore;
  const validator = createStore({ storage: null, now: function () { return 0; } });
  let request = 0;
  let acknowledged = 0;
  let failed = false;
  function parse(text) {
    if (typeof text !== 'string' || !validator.inspectImport(text).ok) return null;
    try { return JSON.parse(text); } catch (error) { return null; }
  }
  function currentText() {
    try { return localStorage.getItem(api.SAVE_KEY); } catch (error) { return null; }
  }
  const native = window.WayfarersNativeCheckpoint;
  const storedText = currentText();
  const stored = parse(storedText);
  const durable = native && parse(native.text);
  // A different guild is replaced only when a previously reviewed import
  // explicitly recorded the identity it superseded. Never replace a newer save.
  if (durable && (!storedText || (stored && durable.savedAt >= stored.savedAt &&
      (durable.state.createdAt === stored.state.createdAt || native.replacesCreatedAt === stored.state.createdAt) &&
      (durable.savedAt > stored.savedAt || durable.state.lastUpdate >= stored.state.lastUpdate)))) {
    try {
      if (storedText && storedText !== native.text) localStorage.setItem(api.BACKUP_KEY, storedText);
      localStorage.setItem(api.SAVE_KEY, native.text);
    } catch (error) { failed = true; }
  }
  delete window.WayfarersNativeCheckpoint;
  function mirror(replacesCreatedAt) {
    const text = currentText();
    if (!parse(text) || !window.WayfarersAndroid) return;
    const message = { type: 'checkpoint', text: text, requestId: ++request };
    if (Number.isFinite(replacesCreatedAt)) message.replacesCreatedAt = replacesCreatedAt;
    window.WayfarersAndroid.postMessage(JSON.stringify(message));
  }
  if (window.WayfarersAndroid) window.WayfarersAndroid.onmessage = function (event) {
    let message;
    try { message = JSON.parse(event.data); } catch (error) { return; }
    if (message.type !== 'checkpoint') return;
    if (message.ok) {
      acknowledged = Math.max(acknowledged, message.requestId);
      if (message.requestId === request) failed = false;
    } else if (message.requestId === request) {
      failed = true;
      const warning = document.querySelector('[data-storage-warning]');
      if (warning) {
        warning.hidden = false;
        warning.textContent = 'Your latest progress could not be saved on this device. Keep the game open and export a backup in Settings.';
      }
    }
  };
  api.createStore = function (options) {
    const store = createStore(options);
    const save = store.save;
    const replaceImport = store.replaceImport;
    store.save = function (state) {
      const result = save(state);
      if (result.ok) mirror();
      return result;
    };
    store.replaceImport = function (text, options) {
      const previous = parse(currentText());
      const result = replaceImport(text, options);
      if (result.ok && result.persisted) mirror(previous && previous.state.createdAt);
      return result;
    };
    return store;
  };
  window.WayfarersCheckpoint = {
    snapshot: function () { const text = currentText(); return parse(text) ? text : ''; },
    confirmed: function () { return request > 0 && acknowledged === request && !failed; }
  };
}());
