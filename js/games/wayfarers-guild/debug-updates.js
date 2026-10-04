(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersDebugUpdates = api;
}(typeof window !== 'undefined' ? window : null, function (root) {
  'use strict';

  const KEY = 'wayfarers-guild-debug-update-reset';
  const identityKeys = ['apkVersion', 'contentVersion'];
  const record = value => value && typeof value === 'object' && !Array.isArray(value);
  const identity = value => record(value) && identityKeys.every(key => Number.isSafeInteger(value[key]) && value[key] > 0);
  const copyIdentity = value => ({ apkVersion: value.apkVersion, contentVersion: value.contentVersion });
  const updateId = value => 'debug-update-apk' + value.apkVersion + '-content' + value.contentVersion;
  const same = (left, right) => !!left && !!right && identityKeys.every(key => left[key] === right[key]);
  const newer = (next, old) => next.apkVersion > old.apkVersion || next.apkVersion === old.apkVersion && next.contentVersion > old.contentVersion;
  const failure = () => ({ ok: false, action: 'none', enabled: false, message: 'The update-reset setting could not be saved. Your guild has been kept.' });

  function createPolicy(options) {
    let storage;
    try { storage = options && Object.prototype.hasOwnProperty.call(options, 'storage') ? options.storage : root && root.localStorage; }
    catch (error) { storage = null; }
    function read() {
      try {
        if (!storage) return failure();
        const raw = storage.getItem(KEY);
        const state = raw ? JSON.parse(raw) : { version: 1, enabled: false, seen: null, pending: null };
        if (!record(state) || state.version !== 1 || typeof state.enabled !== 'boolean' ||
            state.seen !== null && !identity(state.seen) || state.pending !== null &&
            (!identity(state.pending) || state.pending.id !== updateId(state.pending))) return failure();
        return { ok: true, action: 'none', enabled: state.enabled, state, raw, message: '' };
      } catch (error) { return failure(); }
    }
    function write(previous, state, extra) {
      try {
        if (storage.getItem(KEY) !== previous.raw) return failure();
        const text = JSON.stringify(state);
        storage.setItem(KEY, text);
        if (storage.getItem(KEY) !== text) return failure();
        return Object.assign({ ok: true, action: 'none', enabled: state.enabled, state, message: '' }, extra || {});
      } catch (error) { return failure(); }
    }
    function setEnabled(enabled) {
      const previous = read();
      if (!previous.ok || typeof enabled !== 'boolean') return failure();
      return write(previous, Object.assign({}, previous.state, { enabled }));
    }
    function observeCommitted(value, journal) {
      const previous = read();
      if (!previous.ok || !identity(value)) return failure();
      const current = copyIdentity(value), state = previous.state;
      // A restored document explicitly consumes this identity without a reset.
      // First install and enabling the option after an update never erase a guild.
      if (value.recovered === true || !state.seen || !state.enabled) {
        return write(previous, Object.assign({}, state, { seen: current, pending: null }));
      }
      if (!newer(current, state.seen)) return previous;
      const id = updateId(current);
      if (journal && journal.updateId === id && journal.text === null) {
        return write(previous, Object.assign({}, state, { seen: current, pending: null }), { action: 'complete', updateId: id });
      }
      const pending = Object.assign({ id }, current);
      const requested = same(state.pending, current) ? previous : write(previous, Object.assign({}, state, { pending }));
      return requested.ok ? Object.assign({}, requested, { action: 'reset', updateId: id,
        options: { updateId: id, preserveCommerce: true, backupBeforeReset: true } }) : requested;
    }
    function completeReset(id, journal) {
      const previous = read(), pending = previous.state && previous.state.pending;
      if (!previous.ok || !pending || pending.id !== id || !journal || journal.updateId !== id || journal.text !== null) return failure();
      return write(previous, Object.assign({}, previous.state, { seen: copyIdentity(pending), pending: null }), { action: 'complete', updateId: id });
    }
    return { read, setEnabled, observeCommitted, completeReset };
  }

  return { KEY, createPolicy, updateId };
}));
