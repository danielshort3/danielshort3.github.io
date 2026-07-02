(function initProjectStarfallUiCharacterRecord(global) {
  'use strict';

  function clonePlain(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  function ensureAdminConsoleItemInSave(save, options) {
    const settings = options || {};
    const adminConsoleItemId = settings.adminConsoleItemId || 'admin_worldwright_console';
    const getConsumableItem = settings.getConsumableItem || (() => null);
    if (!save || typeof save !== 'object' || !save.state || typeof save.state !== 'object') return save;
    const hasConsoleDefinition = !!getConsumableItem(adminConsoleItemId);
    if (!hasConsoleDefinition) return save;
    const consumables = save.state.consumables && typeof save.state.consumables === 'object' ? save.state.consumables : {};
    save.state.consumables = Object.assign({}, consumables);
    save.state.consumables[adminConsoleItemId] = Math.max(1, Math.floor(Number(save.state.consumables[adminConsoleItemId] || 0) || 0));
    return save;
  }

  function createCharacterRecord(slotId, payload, options) {
    const settings = options || {};
    const copy = settings.clonePlain || clonePlain;
    const normalizeCharacterName = settings.normalizeCharacterName || ((name) => String(name || '').trim().replace(/\s+/g, ' ').slice(0, 18));
    const getClassLabel = settings.getClassLabel || (() => 'Unchosen');
    const getCharacterLook = settings.getCharacterLook || (() => null);
    const getDefaultCharacterLookId = settings.getDefaultCharacterLookId || (() => 'sunlit');
    const nowMs = typeof settings.nowMs === 'function' ? settings.nowMs : Date.now;
    const save = ensureAdminConsoleItemInSave(copy(payload && typeof payload === 'object' ? payload : {}), settings);
    const state = save.state || {};
    const player = state.player || {};
    const classId = String(player.advancedClassId || player.classId || '').trim();
    const baseClassId = String(player.classId || '').trim();
    const name = normalizeCharacterName(player.name) || getClassLabel(baseClassId || classId);
    const look = getCharacterLook(player.lookId);
    return {
      slotId,
      name,
      classId: baseClassId || classId,
      advancedClassId: String(player.advancedClassId || '').trim(),
      lookId: look && look.id || getDefaultCharacterLookId(),
      level: Math.max(1, Math.floor(Number(player.level || 1) || 1)),
      mapId: String(state.mapId || 'starfallCrossing'),
      updatedAt: Number(save.savedAt || nowMs()),
      payload: save
    };
  }

  function createCharacterRecordUiHelpers(options) {
    const settings = options || {};
    return Object.freeze({
      clonePlain,
      ensureAdminConsoleItemInSave: (save) => ensureAdminConsoleItemInSave(save, settings),
      createCharacterRecord: (slotId, payload) => createCharacterRecord(slotId, payload, settings)
    });
  }

  const api = {
    clonePlain,
    ensureAdminConsoleItemInSave,
    createCharacterRecord,
    createCharacterRecordUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.characterRecord = Object.assign({}, modules.characterRecord || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
