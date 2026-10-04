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
  const RESET_KEY = SAVE_KEY + '-reset';
  const FORMAT = 'wayfarers-guild-save';
  const VERSION = 8;
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
    let observedReset = null;
    let resetObserved = false;
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
    function generation() {
      try { return observedReset ? JSON.parse(observedReset).id : ''; } catch (error) { return ''; }
    }
    function saveKey(key, id = generation()) { return id ? key + '-generation-' + id : key; }
    function readSave(key) { return read(saveKey(key)); }
    function writeSave(key, text, requireMirror = false) {
      storage.setItem(saveKey(key), text);
      if (generation()) {
        // A compatibility mirror cannot turn an already committed generation
        // save into a reported failure. Reset itself still requires every old
        // recovery copy to be replaced before its journal can be completed.
        if (requireMirror) storage.setItem(key, text);
        else { try { storage.setItem(key, text); } catch (_) { /* Canonical generation is durable. */ } }
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
        if (![1, 2, 3, 4, 5, 6, 7, VERSION].includes(payload.version)) {
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
        const envelope = { format: FORMAT, version: VERSION, savedAt: time, state: snapshot };
        // Device reset fencing belongs to the local envelope, never the exported
        // guild. A write serialized before a reset cannot impersonate its nonce.
        if (!pretty && observedReset) {
          const reset = parseReset(observedReset);
          if (reset) envelope.resetGeneration = reset.id;
        }
        const text = JSON.stringify(envelope, null, pretty ? 2 : 0);
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
      const reset = read(RESET_KEY);
      if (!reset.ok) { unreadSave = true; return fresh(time, 'unavailable', READ_MESSAGE, false); }
      observedReset = reset.text;
      resetObserved = true;
      if (reset.text) {
        const journal = parseReset(reset.text);
        if (!journal) { unreadSave = true; return fresh(time, 'reset-pending', 'The testing reset record needs recovery. Automatic saving is paused.', false); }
        if (journal.text) {
          const repaired = finishReset(false);
          const pending = parse(journal.text);
          return result(true, 'reset-pending', repaired.ok ? 'Finish the testing reset before continuing.' : repaired.message, {
            state: pending.payload.state, offline: emptyOffline(), canSave: false, savedAt: pending.payload.savedAt
          });
        }
        const mainCandidate = readSave(SAVE_KEY);
        const parsedCandidate = mainCandidate.ok && mainCandidate.text && parse(mainCandidate.text);
        if (!mainCandidate.ok) { unreadSave = true; return fresh(time, 'unavailable', READ_MESSAGE, false); }
        if (!parsedCandidate || !parsedCandidate.ok || parsedCandidate.payload.resetGeneration !== journal.id) {
          const backupCandidate = readSave(BACKUP_KEY);
          const parsedBackup = backupCandidate.ok && backupCandidate.text && parse(backupCandidate.text);
          if (!backupCandidate.ok) { unreadSave = true; return fresh(time, 'unavailable', READ_MESSAGE, false); }
          const recovery = parsedBackup && parsedBackup.ok && parsedBackup.payload.resetGeneration === journal.id ? backupCandidate.text : journal.seedText;
          try { writeSave(BACKUP_KEY, recovery); writeSave(SAVE_KEY, recovery); }
          catch (error) { unreadSave = true; return fresh(time, 'reset-pending', 'Reset recovery is paused. Retry after device storage is available.', false); }
        }
      }
      const deferOffline = !!(loadOptions && loadOptions.deferOffline === true);
      const main = readSave(SAVE_KEY);
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
      const backup = readSave(BACKUP_KEY);
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
      const resetGuard = checkReset();
      if (!resetGuard.ok) return resetGuard;
      const mayReplace = explicitImport || pendingImport;
      if (unreadSave && !mayReplace) return result(false, 'unavailable', READ_MESSAGE);
      if (protectedSave && !mayReplace) return result(false, 'unsupported', UNSUPPORTED_MESSAGE);
      const serialized = serialize(state, now(), false);
      if (!serialized.ok) return serialized;
      const main = readSave(SAVE_KEY);
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
        if (previous && previous.ok) writeSave(BACKUP_KEY, main.text);
        writeSave(SAVE_KEY, serialized.text);
        observedMain = serialized.text;
        hasObservedMain = true;
        const finalGuard = checkReset();
        if (!finalGuard.ok) return Object.assign({}, finalGuard, {
          committed: true,
          message: finalGuard.status === 'unavailable' ? 'Progress was written, but its reset record could not be checked. Retry save before continuing.' : finalGuard.message
        });
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
      const resetGuard = checkReset();
      if (!resetGuard.ok) return resetGuard;
      const parsed = parse(text);
      if (!parsed.ok) return result(false, parsed.status, parsed.message + ' Your current guild has not changed.');
      let restored;
      try {
        restored = restore(parsed.payload, now(), false, options && options.premiumEntitlements);
        // An older backup of this same guild must not reissue consumed practice
        // supplies or help rewards. A deliberately different guild stays separate.
        if (options && options.preservePracticeFrom && core.mergePracticeReceipts) {
          core.mergePracticeReceipts(restored.state, options.preservePracticeFrom);
          if (!validateState(restored.state).ok) throw new Error('Invalid practice receipts.');
        }
      } catch (error) {
        return result(false, 'invalid', 'This guild could not be restored. Your current guild has not changed.');
      }
      pendingImport = true;
      const saved = save(restored.state, true);
      if (saved.ok) protectedSave = false;
      return result(true, saved.ok ? 'imported' : saved.status, saved.ok ? 'Guild imported and saved on this device.' : 'Guild imported for this session. ' + saved.message, Object.assign(restored, { persisted: saved.ok, canSave: true }));
    }

    function parseReset(text) {
      try {
        const value = JSON.parse(text);
        if (!isRecord(value) || value.version !== 1 || !/^[a-zA-Z0-9-]{16,100}$/.test(value.id) ||
          typeof value.previousId !== 'string' || (value.previousId && !/^[a-zA-Z0-9-]{16,100}$/.test(value.previousId)) ||
          (value.previousCreatedAt !== null && !validTime(value.previousCreatedAt)) ||
          (value.text !== null && (typeof value.text !== 'string' || !parse(value.text).ok)) ||
          typeof value.seedText !== 'string' || !parse(value.seedText).ok || parse(value.seedText).payload.resetGeneration !== value.id ||
          (value.text !== null && parse(value.text).payload.resetGeneration !== value.id)) return null;
        return value;
      } catch (error) { return null; }
    }
    function checkReset() {
      const reset = read(RESET_KEY);
      if (!reset.ok) return reset;
      if (!resetObserved) { observedReset = reset.text; resetObserved = true; }
      if (reset.text !== observedReset) return result(false, 'conflict', 'This guild was reset in another window. Reload before continuing.');
      if (reset.text) {
        const journal = parseReset(reset.text);
        if (!journal || journal.text) return result(false, 'reset-pending', 'Finish the testing reset before continuing.');
      }
      return result(true, 'ready');
    }
    function pendingReset() {
      const reset = read(RESET_KEY);
      if (!reset.ok || !reset.text) return null;
      return parseReset(reset.text);
    }
    function finishReset(commit) {
      const raw = read(RESET_KEY);
      if (!raw.ok || raw.text !== observedReset) return result(false, 'conflict', CONFLICT_MESSAGE);
      const journal = raw.text && parseReset(raw.text);
      if (!journal) return result(false, 'reset-pending', 'The reset could not be recovered. Existing files remain protected.');
      if (!journal.text) return result(true, 'reset-complete');
      const parsed = parse(journal.text);
      try {
        // The journal is authoritative until BOTH recovery copies hold the fresh
        // guild. A crash at any write resumes this exact seed, never the old save.
        writeSave(BACKUP_KEY, journal.text, true);
        if (read(RESET_KEY).text !== observedReset) return result(false, 'conflict', CONFLICT_MESSAGE);
        writeSave(SAVE_KEY, journal.text, true);
        observedMain = journal.text;
        hasObservedMain = true;
        if (commit) {
          if (read(RESET_KEY).text !== observedReset) return result(false, 'conflict', CONFLICT_MESSAGE);
          if (journal.previousId && storage.removeItem) {
            storage.removeItem(saveKey(SAVE_KEY, journal.previousId));
            storage.removeItem(saveKey(BACKUP_KEY, journal.previousId));
          }
          const completed = JSON.stringify(Object.assign({}, journal, { text: null }));
          storage.setItem(RESET_KEY, completed);
          observedReset = completed;
          protectedSave = false; unreadSave = false; pendingImport = false;
        }
        return result(true, commit ? 'reset-complete' : 'reset-pending', '', { state: parsed.payload.state, persisted: true, journal });
      } catch (error) { return result(false, 'reset-pending', 'The reset is paused. Retry to finish clearing this device; the same fresh guild will be used.'); }
    }
    function resetForTesting(options) {
      const current = read(RESET_KEY);
      if (!current.ok || (resetObserved && current.text !== observedReset)) return result(false, 'conflict', CONFLICT_MESSAGE);
      const previousJournal = current.text && parseReset(current.text);
      if (current.text && !previousJournal) return result(false, 'reset-pending', 'The reset record could not be read. No progress was changed.');
      if (previousJournal && previousJournal.text) return finishReset(!(options && options.deferCommit));
      const main = readSave(SAVE_KEY);
      if (!main.ok || (hasObservedMain && main.text !== observedMain)) return result(false, 'conflict', CONFLICT_MESSAGE);
      const old = main.text && parse(main.text);
      const time = now();
      const next = core.createState(time);
      if (old && old.ok && next.createdAt === old.payload.state.createdAt) next.createdAt = Math.min(MAX_TIME, next.createdAt + 1);
      const serialized = serialize(next, time, false);
      if (!serialized.ok) return serialized;
      const cryptoApi = settings.crypto || (typeof globalThis !== 'undefined' && globalThis.crypto);
      if (!cryptoApi || typeof cryptoApi.randomUUID !== 'function') return result(false, 'unavailable', 'A secure reset identifier is unavailable. No progress was changed.');
      const id = cryptoApi.randomUUID();
      const freshEnvelope = JSON.parse(serialized.text); freshEnvelope.resetGeneration = id;
      const seedText = JSON.stringify(freshEnvelope);
      const journal = { version: 1, id, previousId: previousJournal ? previousJournal.id : '',
        previousCreatedAt: old && old.ok ? old.payload.state.createdAt : null, text: seedText, seedText };
      try {
        storage.setItem(RESET_KEY, JSON.stringify(journal));
        observedReset = JSON.stringify(journal); resetObserved = true;
      } catch (error) { return result(false, 'unavailable', 'The reset could not be started. Your existing guild is unchanged.'); }
      return finishReset(!(options && options.deferCommit));
    }

    return { load: load, save: function (state) { return save(state, false); }, export: exportSave, inspectImport: inspectImport, replaceImport: replaceImport,
      resetForTesting, pendingReset, finishReset };
  }

  return { createStore: createStore, SAVE_KEY: SAVE_KEY, BACKUP_KEY: BACKUP_KEY, RESET_KEY: RESET_KEY, FORMAT: FORMAT, VERSION: VERSION, MAX_BYTES: MAX_BYTES };
}));
