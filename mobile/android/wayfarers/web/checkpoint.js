(function () {
  'use strict';
  const api = window.WayfarersStorage;
  const createStore = api.createStore;
  const validator = createStore({ storage: null, now: function () { return 0; } });
  let request = 0;
  let acknowledged = 0;
  let failed = false;
  const resetRequests = new Map();
  let generation = '';
  let resetPending = false;
  function parse(text) {
    if (typeof text !== 'string' || !validator.inspectImport(text).ok) return null;
    try { return JSON.parse(text); } catch (error) { return null; }
  }
  function currentText() {
    try { return localStorage.getItem(generation ? api.SAVE_KEY + '-generation-' + generation : api.SAVE_KEY); } catch (error) { return null; }
  }
  const native = window.WayfarersNativeCheckpoint;
  let storedText = currentText();
  let stored = parse(storedText);
  const durable = native && parse(native.text);
  let resetRaw = null, reset = null;
  try {
    resetRaw = localStorage.getItem(api.RESET_KEY);
    reset = resetRaw && JSON.parse(resetRaw);
    if (resetRaw && (!reset || reset.version !== 1 || !/^[a-zA-Z0-9-]{16,100}$/.test(reset.id) ||
      typeof reset.previousId !== 'string' || !parse(reset.seedText) || parse(reset.seedText).resetGeneration !== reset.id ||
      (reset.text !== null && !parse(reset.text)))) throw new Error('Invalid reset journal');
    generation = reset ? reset.id : '';
    resetPending = !!(reset && reset.text);
    const nativeGeneration = native && native.generation || '';
    if (durable && nativeGeneration && generation !== nativeGeneration &&
      (!reset || (!resetPending && native.previousGeneration === generation))) {
      // Native reset committed before WebView files did. Never retain the erased
      // guild as a fallback when restoring a new reset generation.
      localStorage.setItem(api.RESET_KEY, JSON.stringify({ version: 1, id: nativeGeneration, previousId: native.previousGeneration || '', previousCreatedAt: native.replacesCreatedAt ?? null, text: native.text, seedText: native.text }));
      generation = nativeGeneration; resetPending = true;
    } else if (durable && resetPending && nativeGeneration === generation &&
      durable.state.createdAt === parse(reset.text).state.createdAt && durable.savedAt >= parse(reset.text).savedAt) {
      localStorage.setItem(api.RESET_KEY, JSON.stringify(Object.assign({}, reset, { text: native.text })));
    } else if (durable && generation !== nativeGeneration && !(resetPending && reset.previousId === nativeGeneration)) {
      failed = true;
    }
  } catch (error) { failed = true; resetPending = true; }
  storedText = currentText();
  stored = parse(storedText);
  // A different guild is replaced only when a previously reviewed import
  // explicitly recorded the identity it superseded. Never replace a newer save.
  if (!failed && !resetPending && durable && (!storedText || (stored && durable.savedAt >= stored.savedAt &&
      (durable.state.createdAt === stored.state.createdAt || native.replacesCreatedAt === stored.state.createdAt) &&
      (durable.savedAt > stored.savedAt || durable.state.lastUpdate >= stored.state.lastUpdate)))) {
    try {
      if (storedText && storedText !== native.text) localStorage.setItem(api.BACKUP_KEY, storedText);
      localStorage.setItem(api.SAVE_KEY, native.text);
      if (generation) {
        const key = api.SAVE_KEY + '-generation-' + generation;
        const current = localStorage.getItem(key);
        if (current && current !== native.text) localStorage.setItem(api.BACKUP_KEY + '-generation-' + generation, current);
        localStorage.setItem(key, native.text);
      }
    } catch (error) { failed = true; }
  }
  delete window.WayfarersNativeCheckpoint;
  function mirror(replacesCreatedAt) {
    const text = currentText();
    if (resetPending || !parse(text) || !window.WayfarersAndroid) return;
    const message = { type: 'checkpoint', text: text, generation: generation, requestId: ++request,
      documentToken: window.WayfarersContent && window.WayfarersContent.documentToken };
    if (Number.isFinite(replacesCreatedAt)) message.replacesCreatedAt = replacesCreatedAt;
    window.WayfarersAndroid.postMessage(JSON.stringify(message));
  }
  if (window.WayfarersAndroid) window.WayfarersAndroid.onmessage = function (event) {
    let message;
    try { message = JSON.parse(event.data); } catch (error) { return; }
    if (message.type === 'reset-guild') {
      const pending = resetRequests.get(message.requestId);
      if (pending) { resetRequests.delete(message.requestId); clearTimeout(pending.timer); pending.resolve(message.ok === true); }
      return;
    }
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
    const resetForTesting = store.resetForTesting;
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
    store.resetForTesting = async function () {
      const result = resetForTesting({ deferCommit: true });
      if (!result.ok) return result;
      const journal = result.journal || store.pendingReset();
      if (!journal || !journal.text) return { ok: false, status: 'reset-pending', message: 'The reset could not be verified.' };
      resetPending = true;
      generation = journal.id;
      if (!window.WayfarersAndroid) return { ok: false, status: 'reset-pending', message: 'Reconnect Android save storage, then retry the reset.' };
      const requestId = ++request;
      const confirmed = await new Promise(resolve => {
        const timer = setTimeout(() => { resetRequests.delete(requestId); resolve(false); }, 8000);
        resetRequests.set(requestId, { resolve, timer });
        try { window.WayfarersAndroid.postMessage(JSON.stringify({ type: 'reset-guild', requestId, text: journal.text, generation: journal.id, previousGeneration: journal.previousId,
          documentToken: window.WayfarersContent && window.WayfarersContent.documentToken })); }
        catch (error) { clearTimeout(timer); resetRequests.delete(requestId); resolve(false); }
      });
      if (!confirmed) { failed = true; return { ok: false, status: 'reset-pending', message: 'Android has not confirmed the reset. Retry to finish; no new guild will be rolled.' }; }
      const committed = store.finishReset(true);
      if (committed.ok) { resetPending = false; failed = false; acknowledged = request; }
      return committed;
    };
    return store;
  };
  window.WayfarersCheckpoint = {
    snapshot: function () { const text = currentText(); return !resetPending && parse(text) ? text : ''; },
    generation: function () { return generation; },
    confirmed: function () { return !resetPending && request > 0 && acknowledged === request && !failed; }
  };
}());
