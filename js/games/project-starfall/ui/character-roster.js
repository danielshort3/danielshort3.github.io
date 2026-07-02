(function initProjectStarfallUiCharacterRoster(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const CHARACTER_ROSTER_KEY = 'projectStarfallCharacterRoster.v1';
  const CHARACTER_SLOT_COUNT = 8;
  const CHARACTER_SELECT_PAGE_SIZE = 4;
  const DOM_CHARACTER_CREATE_CONFIRM_ATTRIBUTES = Object.freeze([
    'data-starfall-character-create-confirm'
  ]);
  const DOM_CHARACTER_CREATE_CONFIRM_SELECTOR = DOM_CHARACTER_CREATE_CONFIRM_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function getCharacterLooks(options) {
    const data = options && options.data || Data;
    return Array.isArray(data.CHARACTER_LOOKS) && data.CHARACTER_LOOKS.length
      ? data.CHARACTER_LOOKS
      : [{ id: 'sunlit', name: 'Sunlit', shirt: '#526f86', shirtLight: '#8caabd', pants: '#33495b', hair: '#3b2725', skin: '#d99a6c' }];
  }

  function getDefaultCharacterLookId(options) {
    const first = getCharacterLooks(options)[0];
    return first && first.id || 'sunlit';
  }

  function getCharacterLook(lookId, options) {
    const id = String(lookId || '').trim();
    const looks = getCharacterLooks(options);
    return looks.find((look) => look && look.id === id) || looks[0];
  }

  function normalizeCharacterName(name) {
    return String(name || '').trim().replace(/\s+/g, ' ').slice(0, 18);
  }

  function getCharacterSlotId(index) {
    return `slot_${Math.max(1, Math.floor(Number(index || 0) || 0) + 1)}`;
  }

  function createEmptyCharacterSlot(index) {
    return { slotId: getCharacterSlotId(index), index, character: null };
  }

  function getClassLabel(classId, options) {
    const data = options && options.data || Data;
    const advancedClasses = data.ADVANCED_CLASSES || {};
    const baseClasses = data.BASE_CLASSES || {};
    const classData = advancedClasses[classId] || baseClasses[classId];
    return classData ? classData.name : 'Unchosen';
  }

  function getMapLabel(mapId, options) {
    const data = options && options.data || Data;
    const map = (data.MAPS || []).find((candidate) => candidate && candidate.id === mapId);
    return map ? map.name : 'Unknown Map';
  }

  function formatCharacterSavedAt(value, options) {
    const settings = options || {};
    const formatMetricEta = settings.formatMetricEta || ((seconds) => `${Math.ceil(Number(seconds) || 0)}s`);
    const nowMs = typeof settings.nowMs === 'function' ? settings.nowMs() : Date.now();
    const savedAt = Number(value || 0);
    if (!savedAt || !Number.isFinite(savedAt)) return 'Unknown';
    const elapsed = Math.max(0, (nowMs - savedAt) / 1000);
    if (elapsed < 60) return 'Just now';
    if (elapsed < 60 * 60 * 24 * 30) return `${formatMetricEta(elapsed)} ago`;
    const date = new Date(savedAt);
    if (!Number.isFinite(date.getTime())) return 'Unknown';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function getCharacterCreateConfirmButton(panel, selector) {
    const source = panel || null;
    const querySelector = selector || DOM_CHARACTER_CREATE_CONFIRM_SELECTOR;
    return source && typeof source.querySelector === 'function'
      ? source.querySelector(querySelector)
      : null;
  }

  function getCharacterRosterStorageKey(options) {
    const data = options && options.data || Data;
    return data.CHARACTER_ROSTER_KEY || CHARACTER_ROSTER_KEY;
  }

  function getCharacterSlotCount(options) {
    const data = options && options.data || Data;
    return Math.max(1, Math.floor(Number(data.CHARACTER_SLOT_COUNT || CHARACTER_SLOT_COUNT) || CHARACTER_SLOT_COUNT));
  }

  function getCharacterSelectPageSize(value) {
    return Math.max(1, Math.floor(Number(value || CHARACTER_SELECT_PAGE_SIZE) || CHARACTER_SELECT_PAGE_SIZE));
  }

  function getCharacterSelectDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const pageAction = getAttribute('data-starfall-character-page');
    if (pageAction) return { handled: true, type: 'page', direction: pageAction };
    const slotId = getAttribute('data-starfall-character-slot');
    if (slotId) return { handled: true, type: 'selectSlot', slotId };
    if (hasAttribute('data-starfall-character-start')) return { handled: true, type: 'start' };
    const createSlotId = getAttribute('data-starfall-character-create-open');
    if (createSlotId) return { handled: true, type: 'openCreate', slotId: createSlotId };
    if (hasAttribute('data-starfall-character-create-confirm')) return { handled: true, type: 'confirmCreate' };
    if (hasAttribute('data-starfall-character-create-cancel')) return { handled: true, type: 'cancelCreate' };
    if (hasAttribute('data-starfall-character-delete-cancel')) return { handled: true, type: 'cancelDelete' };
    if (hasAttribute('data-starfall-character-delete')) return { handled: true, type: 'delete' };
    const classId = getAttribute('data-starfall-character-class');
    if (classId) return { handled: true, type: 'chooseClass', classId };
    const lookId = getAttribute('data-starfall-character-look');
    if (lookId) return { handled: true, type: 'chooseLook', lookId };
    return { handled: false, type: '' };
  }

  function getBaseClassSelectDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const classId = getAttribute('data-starfall-class');
    if (classId) return { handled: true, type: 'chooseBaseClass', classId };
    return { handled: false, type: '' };
  }

  function getCharacterNameInputDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (!hasAttribute('data-starfall-character-name')) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'setCharacterNameDraft',
      value: source && source.value
    };
  }

  function getCharacterRosterRegionAction(region) {
    const source = region || {};
    if (source.type === 'roster-trait') return { handled: true, type: 'toggleRosterTrait', traitId: source.traitId };
    return { handled: false, type: '' };
  }

  function createCharacterRosterUiHelpers(options) {
    const settings = options || {};
    return Object.freeze({
      getCharacterLooks: () => getCharacterLooks(settings),
      getDefaultCharacterLookId: () => getDefaultCharacterLookId(settings),
      getCharacterLook: (lookId) => getCharacterLook(lookId, settings),
      normalizeCharacterName,
      getCharacterSlotId,
      createEmptyCharacterSlot,
      getClassLabel: (classId) => getClassLabel(classId, settings),
      getMapLabel: (mapId) => getMapLabel(mapId, settings),
      formatCharacterSavedAt: (value) => formatCharacterSavedAt(value, settings),
      getCharacterCreateConfirmButton,
      getCharacterRosterStorageKey: () => getCharacterRosterStorageKey(settings),
      getCharacterSlotCount: () => getCharacterSlotCount(settings),
      getCharacterSelectPageSize,
      getCharacterSelectDomAction,
      getBaseClassSelectDomAction,
      getCharacterNameInputDomAction,
      getCharacterRosterRegionAction
    });
  }

  function createCharacterRosterDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_CHARACTER_CREATE_CONFIRM_ATTRIBUTES,
      DOM_CHARACTER_CREATE_CONFIRM_SELECTOR
    });
  }

  const api = {
    CHARACTER_ROSTER_KEY,
    CHARACTER_SLOT_COUNT,
    CHARACTER_SELECT_PAGE_SIZE,
    DOM_CHARACTER_CREATE_CONFIRM_ATTRIBUTES,
    DOM_CHARACTER_CREATE_CONFIRM_SELECTOR,
    getCharacterLooks,
    getDefaultCharacterLookId,
    getCharacterLook,
    normalizeCharacterName,
    getCharacterSlotId,
    createEmptyCharacterSlot,
    getClassLabel,
    getMapLabel,
    formatCharacterSavedAt,
    getCharacterCreateConfirmButton,
    getCharacterRosterStorageKey,
    getCharacterSlotCount,
    getCharacterSelectPageSize,
    getCharacterSelectDomAction,
    getBaseClassSelectDomAction,
    getCharacterNameInputDomAction,
    getCharacterRosterRegionAction,
    createCharacterRosterUiHelpers,
    createCharacterRosterDomSelectorUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.characterRoster = Object.assign({}, modules.characterRoster || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
