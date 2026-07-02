(function initProjectStarfallUiItemRequirements(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getFallbackClassLabel(classId) {
    return String(classId || '')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }

  function escapeHtmlFallback(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function playerMeetsItemLevel(snapshot, item) {
    const player = snapshot && snapshot.state ? snapshot.state.player || {} : {};
    return Number(player.level || 1) >= Number(item && item.level || 1);
  }

  function getItemClassIds(item, options) {
    const normalize = options && options.normalizeId || normalizeId;
    const ids = Array.isArray(item && item.classIds)
      ? item.classIds.map(normalize).filter(Boolean)
      : [];
    const single = normalize(item && item.classId || 'any');
    if (!ids.length && single) ids.push(single);
    return ids.length ? Array.from(new Set(ids)) : ['any'];
  }

  function itemMatchesPlayerClass(item, player, options) {
    const normalize = options && options.normalizeId || normalizeId;
    const ids = getItemClassIds(item, options);
    if (ids.includes('any')) return true;
    const playerIds = new Set([
      normalize(player && player.classId),
      normalize(player && player.advancedClassId)
    ].filter(Boolean));
    return ids.some((id) => playerIds.has(id));
  }

  function getItemClassRequirementLabel(item, options) {
    const getClassLabel = options && options.getClassLabel || getFallbackClassLabel;
    const ids = getItemClassIds(item, options);
    if (ids.includes('any')) return 'Any class';
    return ids.map((id) => getClassLabel(id)).join(', ');
  }

  function getItemClassBadgeLabel(item, options) {
    return getItemClassRequirementLabel(item, options).replace(/\s+class$/i, '');
  }

  function isItemClassBlockedForPlayer(item, player, options) {
    if (!item || !player || !player.classId) return false;
    return !itemMatchesPlayerClass(item, player, options);
  }

  function isItemClassBlockedForSnapshot(snapshot, item, options) {
    const player = snapshot && snapshot.state ? snapshot.state.player || {} : {};
    return isItemClassBlockedForPlayer(item, player, options);
  }

  function isItemRequirementBlockedForSnapshot(snapshot, item, options) {
    return isItemClassBlockedForSnapshot(snapshot, item, options) || !playerMeetsItemLevel(snapshot, item);
  }

  function renderItemLevelBadge(snapshot, item, options) {
    if (!item) return '';
    const tone = playerMeetsItemLevel(snapshot, item) ? 'is-met' : 'is-low';
    return `<span class="project-starfall-inventory-level ${tone}" aria-hidden="true">Lv ${Number(item.level || 1)}</span>`;
  }

  function renderItemClassBadge(item, snapshot, options) {
    if (!item || !item.slot) return '';
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const blocked = isItemClassBlockedForSnapshot(snapshot, item, settings);
    return `<span class="project-starfall-item-class-badge ${blocked ? 'is-blocked' : ''}" aria-hidden="true">${escapeHtml(getItemClassBadgeLabel(item, settings))}</span>`;
  }

  function createItemRequirementUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      getClassLabel: settings.getClassLabel || getFallbackClassLabel,
      normalizeId: settings.normalizeId || normalizeId,
      escapeHtml: settings.escapeHtml || escapeHtmlFallback
    });
    const mergeOptions = (override) => Object.assign({}, helperOptions, override || {});
    return Object.freeze({
      playerMeetsItemLevel,
      getItemClassIds: (item, override) => getItemClassIds(item, mergeOptions(override)),
      itemMatchesPlayerClass: (item, player, override) => itemMatchesPlayerClass(item, player, mergeOptions(override)),
      getItemClassRequirementLabel: (item, override) => getItemClassRequirementLabel(item, mergeOptions(override)),
      getItemClassBadgeLabel: (item, override) => getItemClassBadgeLabel(item, mergeOptions(override)),
      isItemClassBlockedForPlayer: (item, player, override) => isItemClassBlockedForPlayer(item, player, mergeOptions(override)),
      isItemClassBlockedForSnapshot: (snapshot, item, override) => isItemClassBlockedForSnapshot(snapshot, item, mergeOptions(override)),
      isItemRequirementBlockedForSnapshot: (snapshot, item, override) => isItemRequirementBlockedForSnapshot(snapshot, item, mergeOptions(override)),
      renderItemLevelBadge: (snapshot, item, override) => renderItemLevelBadge(snapshot, item, mergeOptions(override)),
      renderItemClassBadge: (item, snapshot, override) => renderItemClassBadge(item, snapshot, mergeOptions(override))
    });
  }

  const api = {
    playerMeetsItemLevel,
    getItemClassIds,
    itemMatchesPlayerClass,
    getItemClassRequirementLabel,
    getItemClassBadgeLabel,
    isItemClassBlockedForPlayer,
    isItemClassBlockedForSnapshot,
    isItemRequirementBlockedForSnapshot,
    renderItemLevelBadge,
    renderItemClassBadge,
    createItemRequirementUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.itemRequirements = Object.assign({}, modules.itemRequirements || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
