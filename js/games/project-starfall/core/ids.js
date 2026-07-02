(function initProjectStarfallCoreIds(global) {
  'use strict';

  const getByIdCache = new WeakMap();

  function normalizeId(value) {
    return String(value || '').trim();
  }

  function getById(items, id) {
    const source = Array.isArray(items) ? items : [];
    if (!source.length) return null;
    const first = source[0] || null;
    const last = source[source.length - 1] || null;
    const cache = getByIdCache.get(source);
    if (cache &&
      cache.length === source.length &&
      cache.first === first &&
      cache.last === last) {
      return cache.map.has(id) ? cache.map.get(id) : null;
    }
    const map = new Map();
    for (let index = 0; index < source.length; index += 1) {
      const item = source[index];
      if (item && !map.has(item.id)) map.set(item.id, item);
    }
    getByIdCache.set(source, {
      length: source.length,
      first,
      last,
      map
    });
    return map.has(id) ? map.get(id) : null;
  }

  const api = {
    getById,
    getByIdCache,
    normalizeId
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
