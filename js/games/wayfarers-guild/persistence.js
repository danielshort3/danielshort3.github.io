(function (root, factory) {
  const api = factory(typeof module === 'object' && module.exports ? require('./core.js') : root.WayfarersCore, root);
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root) root.WayfarersStorage = api;
}(typeof window !== 'undefined' ? window : null, function (defaultCore, root) {
  'use strict';

  const SAVE_KEY = 'wayfarers-guild-save-v1';
  const BACKUP_KEY = SAVE_KEY + '-backup';
  const FORMAT = 'wayfarers-guild-save';
  const VERSION = 6;
  const MAX_BYTES = 1024 * 1024;
  const MAX_TIME = 8.64e15;
  const STORAGE_MESSAGE = 'Progress is safe in this open game, but this browser could not save it. Export a save in Settings before closing, then allow browser storage or free some space.';
  const READ_MESSAGE = 'This browser could not read local saves. Automatic saving is paused to protect any existing guild. Reload to try again. Export a save in Settings to keep this session.';
  const CONFLICT_MESSAGE = 'Another tab changed this guild’s save. Automatic saving is paused to protect that progress. Reload to use the latest save, or export this session in Settings before choosing which guild to keep.';
  const UNSUPPORTED_MESSAGE = 'This save uses an unsupported Wayfarers version. Update the app or import a supported save. Your existing save has been kept.';

  function result(ok, status, message, extra) {
    return Object.assign({ ok: ok, status: status, message: message || '' }, extra || {});
  }

  function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function validTime(value) {
    return Number.isFinite(value) && value >= 0 && value <= MAX_TIME;
  }

  function byteLength(value) {
    let bytes = 0;
    for (const char of value) {
      const code = char.codePointAt(0);
      bytes += code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
      if (bytes >= MAX_BYTES) break;
    }
    return bytes;
  }

  function emptyOffline() {
    return { seconds: 0, elapsedSeconds: 0, pendingSeconds: 0, clockSkew: false, gains: {}, events: [] };
  }

  function createStore(options) {
    const settings = options || {};
    const core = settings.core || defaultCore;
    const clock = typeof settings.now === 'function' ? settings.now : Date.now;
    let storage = null;
    let protectedSave = false;
    let unreadSave = false;
    let pendingImport = false;
    let observedMain = null;
    let hasObservedMain = false;
    try {
      storage = Object.prototype.hasOwnProperty.call(settings, 'storage') ? settings.storage : root && root.localStorage;
    } catch (error) {
      storage = null;
    }

    function now() {
      try {
        const time = clock();
        return validTime(time) ? time : Date.now();
      } catch (error) {
        return Date.now();
      }
    }

    function read(key) {
      try {
        if (!storage || typeof storage.getItem !== 'function') return result(false, 'unavailable', STORAGE_MESSAGE);
        return result(true, 'read', '', { text: storage.getItem(key) });
      } catch (error) {
        return result(false, 'unavailable', STORAGE_MESSAGE);
      }
    }

    function validateState(state) {
      try {
        if (!isRecord(state)) return result(false, 'invalid', 'The save does not contain a Wayfarers guild.');
        if (state.schemaVersion !== VERSION) {
          return result(false, typeof state.schemaVersion === 'number' ? 'unsupported' : 'invalid', 'The guild has an unsupported or missing schema version.');
        }
        const validation = core.validateState(state);
        if (!validation || !validation.valid) return result(false, 'invalid', 'The save contains incomplete or invalid guild data.');
        return result(true, 'valid');
      } catch (error) {
        return result(false, 'invalid', 'The save contains invalid guild data.');
      }
    }

    function parse(text) {
      if (typeof text !== 'string') return result(false, 'invalid', 'Choose a Wayfarers save file.');
      if (text.length >= MAX_BYTES || byteLength(text) >= MAX_BYTES) return result(false, 'invalid', 'The save must be smaller than 1 MB.');
      if (!text.trim()) return result(false, 'invalid', 'Choose a Wayfarers save file.');
      try {
        const payload = JSON.parse(text, function (key, value) {
          if (key === '__proto__' || key === 'prototype' || key === 'constructor') throw new Error('Unsafe property');
          return value;
        });
        if (!isRecord(payload) || payload.format !== FORMAT) return result(false, 'invalid', 'This file is not a Wayfarers save.');
        if (![1, 2, 3, 4, 5, VERSION].includes(payload.version)) {
          return result(false, typeof payload.version === 'number' ? 'unsupported' : 'invalid', UNSUPPORTED_MESSAGE);
        }
        if (!validTime(payload.savedAt)) return result(false, 'invalid', 'The save is missing a valid save date.');
        if (payload.version !== VERSION) {
          // Only the actual legacy schema is migrated. A mislabeled future save must
          // remain protected, and failed migration must never become a fresh guild.
          if (!isRecord(payload.state) || payload.state.schemaVersion !== payload.version) return result(false, 'unsupported', UNSUPPORTED_MESSAGE);
          const migrated = typeof core.migrateState === 'function' ? core.migrateState(payload.state) : null;
          if (!migrated) return result(false, 'invalid', 'This legacy save contains incomplete or invalid guild data.');
          payload.state = migrated;
          payload.version = VERSION;
        }
        const validation = validateState(payload.state);
        if (!validation.ok) return validation;
        return result(true, 'valid', '', { payload: payload });
      } catch (error) {
        return result(false, 'invalid', 'The file could not be read as a valid Wayfarers save.');
      }
    }

    function serialize(state, time, pretty) {
      const validation = validateState(state);
      if (!validation.ok) return validation;
      try {
        // Normalize a detached snapshot; saving must not advance or mutate the live guild.
        const snapshot = core.normalizeState(JSON.parse(JSON.stringify(state)), time);
        const checked = validateState(snapshot);
        if (!checked.ok) return checked;
        const text = JSON.stringify({ format: FORMAT, version: VERSION, savedAt: time, state: snapshot }, null, pretty ? 2 : 0);
        if (text.length >= MAX_BYTES || byteLength(text) >= MAX_BYTES) return result(false, 'invalid', 'The save must be smaller than 1 MB.');
        return result(true, 'serialized', '', { text: text });
      } catch (error) {
        return result(false, 'invalid', 'This guild could not be saved. Keep the game open and try exporting again.');
      }
    }

    function restore(payload, time, deferOffline, premiumEntitlements) {
      const state = core.normalizeState(payload.state, time);
      if (Array.isArray(premiumEntitlements)) core.setPremiumEntitlements(state, premiumEntitlements);
      // Native Play ownership is a non-exportable engine cache. The host may need
      // to restore that cache before deciding the rates for the elapsed interval.
      if (deferOffline) return { state: state, offline: emptyOffline(), savedAt: payload.savedAt, deferredOffline: true };
      // lastUpdate is the simulation timestamp. savedAt records the file write, not earned time.
      // The core shares this exact calculation with foreground play and never truncates an absence.
      const elapsedSeconds = Math.max(0, (time - state.lastUpdate) / 1000);
      const clockSkew = time < state.lastUpdate;
      const advanced = core.advanceTo(state, time);
      if (!validateState(state).ok) throw new Error('Advancement produced invalid state');
      return {
        state: state,
        offline: {
          seconds: advanced.seconds || 0,
          elapsedSeconds: elapsedSeconds,
          pendingSeconds: advanced.pendingSeconds || 0,
          clockSkew: clockSkew,
          gains: advanced.gained || {},
          events: advanced.events || [],
          summary: advanced.summary || null
        },
        savedAt: payload.savedAt
      };
    }

    function fresh(time, status, message, canSave) {
      return result(true, status, message, {
        state: core.createState(time),
        offline: emptyOffline(),
        canSave: canSave,
        savedAt: null
      });
    }

    function load(loadOptions) {
      const time = now();
      const deferOffline = !!(loadOptions && loadOptions.deferOffline === true);
      const main = read(SAVE_KEY);
      protectedSave = false;
      unreadSave = false;
      pendingImport = false;
      hasObservedMain = false;
      if (!main.ok) {
        unreadSave = true;
        return fresh(time, 'unavailable', READ_MESSAGE, false);
      }
      observedMain = main.text;
      hasObservedMain = true;
      const parsed = main.text === null ? null : parse(main.text);
      if (parsed && parsed.ok) {
        try {
          return result(true, 'loaded', '', Object.assign(restore(parsed.payload, time, deferOffline), { canSave: true }));
        } catch (error) {
          unreadSave = true;
          return fresh(time, 'unavailable', 'The saved guild could not safely resume. Its data has been kept and automatic saving is paused. Reload to try again or import an exported save.', false);
        }
      }
      if (parsed && parsed.status === 'unsupported') {
        protectedSave = true;
        return fresh(time, 'unsupported', UNSUPPORTED_MESSAGE, false);
      }
      const backup = read(BACKUP_KEY);
      if (!backup.ok) {
        unreadSave = true;
        return fresh(time, 'unavailable', READ_MESSAGE, false);
      }
      const recovered = backup.text === null ? null : parse(backup.text);
      if (recovered && recovered.ok) {
        try {
          return result(true, 'recovered', 'Your last valid backup was recovered. Some of the most recent changes may be missing.', Object.assign(restore(recovered.payload, time, deferOffline), { canSave: true }));
        } catch (error) {
          unreadSave = true;
          return fresh(time, 'unavailable', 'The backup could not safely resume. Its data has been kept and automatic saving is paused. Reload to try again or import an exported save.', false);
        }
      }
      if (recovered && recovered.status === 'unsupported') {
        protectedSave = true;
        return fresh(time, 'unsupported', UNSUPPORTED_MESSAGE, false);
      }
      if (parsed || recovered) {
        unreadSave = true;
        return fresh(time, 'invalid', 'The local save and backup could not be read. Their original bytes are protected and automatic saving is paused. Import a reviewed exported save to recover your guild; this temporary session can also be exported.', false);
      }
      return fresh(time, 'new', '', true);
    }

    function save(state, explicitImport) {
      const mayReplace = explicitImport || pendingImport;
      if (unreadSave && !mayReplace) return result(false, 'unavailable', READ_MESSAGE);
      if (protectedSave && !mayReplace) return result(false, 'unsupported', UNSUPPORTED_MESSAGE);
      const serialized = serialize(state, now(), false);
      if (!serialized.ok) return serialized;
      const main = read(SAVE_KEY);
      if (!main.ok) return main;
      if (hasObservedMain && main.text !== observedMain && !explicitImport) return result(false, 'conflict', CONFLICT_MESSAGE);
      const previous = main.text === null ? null : parse(main.text);
      if (previous && previous.status === 'unsupported' && !mayReplace) {
        protectedSave = true;
        return result(false, 'unsupported', UNSUPPORTED_MESSAGE);
      }
      if (explicitImport) {
        observedMain = main.text;
        hasObservedMain = true;
      }
      try {
        if (!storage || typeof storage.setItem !== 'function') return result(false, 'unavailable', STORAGE_MESSAGE);
        // A corrupt main must never replace a good backup. A failed backup write also leaves
        // the main untouched, so quota/security failures cannot sacrifice the last good save.
        if (previous && previous.ok) storage.setItem(BACKUP_KEY, main.text);
        storage.setItem(SAVE_KEY, serialized.text);
        observedMain = serialized.text;
        hasObservedMain = true;
        protectedSave = false;
        unreadSave = false;
        pendingImport = false;
        return result(true, 'saved', 'Guild saved on this device.');
      } catch (error) {
        return result(false, 'unavailable', STORAGE_MESSAGE);
      }
    }

    function exportSave(state) {
      const time = now();
      const serialized = serialize(state, time, true);
      if (!serialized.ok) return serialized;
      return result(true, 'exported', '', {
        text: serialized.text,
        filename: 'wayfarers-guild-save-' + new Date(time).toISOString().slice(0, 10) + '.json'
      });
    }

    function inspectImport(text, options) {
      const parsed = parse(text);
      if (!parsed.ok) return result(false, parsed.status, parsed.message + ' Your current guild has not changed.');
      try {
        return result(true, 'valid', '', restore(parsed.payload, now(), false, options && options.premiumEntitlements));
      } catch (error) {
        return result(false, 'invalid', 'This guild could not be restored. Your current guild has not changed.');
      }
    }

    function replaceImport(text, options) {
      const parsed = parse(text);
      if (!parsed.ok) return result(false, parsed.status, parsed.message + ' Your current guild has not changed.');
      let restored;
      try {
        restored = restore(parsed.payload, now(), false, options && options.premiumEntitlements);
      } catch (error) {
        return result(false, 'invalid', 'This guild could not be restored. Your current guild has not changed.');
      }
      pendingImport = true;
      const saved = save(restored.state, true);
      if (saved.ok) protectedSave = false;
      return result(true, saved.ok ? 'imported' : saved.status, saved.ok ? 'Guild imported and saved on this device.' : 'Guild imported for this session. ' + saved.message, Object.assign(restored, { persisted: saved.ok, canSave: true }));
    }

    return { load: load, save: function (state) { return save(state, false); }, export: exportSave, inspectImport: inspectImport, replaceImport: replaceImport };
  }

  return { createStore: createStore, SAVE_KEY: SAVE_KEY, BACKUP_KEY: BACKUP_KEY, FORMAT: FORMAT, VERSION: VERSION, MAX_BYTES: MAX_BYTES };
}));
