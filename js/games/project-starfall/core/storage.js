(function initProjectStarfallCoreStorage(global) {
  'use strict';

  function getLocalStorage() {
    return typeof localStorage === 'undefined' ? null : localStorage;
  }

  function isStorageAvailable() {
    return !!getLocalStorage();
  }

  function readStorageText(key) {
    const storage = getLocalStorage();
    return storage ? storage.getItem(key) : null;
  }

  function writeStorageText(key, value) {
    const storage = getLocalStorage();
    if (!storage) return false;
    storage.setItem(key, String(value));
    return true;
  }

  function removeStorageItem(key) {
    const storage = getLocalStorage();
    if (!storage) return false;
    storage.removeItem(key);
    return true;
  }

  function readStorageJson(key, fallback) {
    const raw = readStorageText(key);
    return raw ? JSON.parse(raw) : fallback;
  }

  function writeStorageJson(key, value) {
    return writeStorageText(key, JSON.stringify(value));
  }

  const api = {
    isStorageAvailable,
    readStorageText,
    writeStorageText,
    removeStorageItem,
    readStorageJson,
    writeStorageJson
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
