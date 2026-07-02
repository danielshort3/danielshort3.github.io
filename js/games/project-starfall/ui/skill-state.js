(function initProjectStarfallUiSkillState(global) {
  'use strict';

  const UiModules = global.ProjectStarfallUiModules || {};
  const UiLookups = (typeof require === 'function' ? require('./lookups.js') : null) || UiModules.lookups || {};
  const UiSkillMetadata = (typeof require === 'function' ? require('./skill-metadata.js') : null) || UiModules.skillMetadata || {};
  const UiSkillPrerequisites = (typeof require === 'function' ? require('./skill-prerequisites.js') : null) || UiModules.skillPrerequisites || {};

  function getSkillById(skillId, options) {
    const settings = options || {};
    if (settings.getSkillById) return settings.getSkillById(skillId);
    if (UiLookups.getSkillById) return UiLookups.getSkillById(skillId, settings);
    return null;
  }

  function isSkillUnlocked(snapshot, skill) {
    if (!skill || !skill.prerequisites || !skill.prerequisites.length) return true;
    const ranks = snapshot.state.skills || {};
    const groupedAny = skill.prerequisites.filter((item) => item.any);
    const strict = skill.prerequisites.filter((item) => !item.any);
    const hasRank = (id, rank) => Number(ranks[id] || 0) >= Number(rank || 1);
    const meets = (item) => {
      if (Array.isArray(item.anyOf)) return item.anyOf.some((id) => hasRank(id, item.rank));
      return hasRank(item.skillId, item.rank);
    };
    if (strict.some((item) => !meets(item))) return false;
    if (groupedAny.length) return groupedAny.some((item) => meets(item));
    return true;
  }

  function describePrerequisites(skill, options) {
    if (!skill || !skill.prerequisites || !skill.prerequisites.length) return 'Open immediately';
    const any = skill.prerequisites.filter((item) => item.any);
    const strict = skill.prerequisites.filter((item) => !item.any);
    const parts = strict.map((item) => {
      if (Array.isArray(item.anyOf)) return `Any listed skill level ${item.rank}`;
      const prereq = getSkillById(item.skillId, options);
      return `${prereq ? prereq.name : item.skillId} level ${item.rank}`;
    });
    if (any.length) {
      const labels = any.map((item) => {
        const prereq = getSkillById(item.skillId, options);
        return `${prereq ? prereq.name : item.skillId} level ${item.rank}`;
      });
      parts.push(`One of: ${labels.join(' or ')}`);
    }
    return parts.join('; ');
  }

  function getSkillUiState(snapshot, skill, options) {
    const settings = options || {};
    const getSkillPointPoolId = settings.getSkillPointPoolId || UiSkillMetadata.getSkillPointPoolId;
    const isPassiveSkill = settings.isPassiveSkill || UiSkillMetadata.isPassiveSkill || (() => false);
    const getMissingPrerequisiteLabels = settings.getMissingPrerequisiteLabels || UiSkillPrerequisites.getMissingPrerequisiteLabels || (() => []);
    const rank = Number(snapshot.state.skills[skill.id] || 0);
    const unlocked = isSkillUnlocked(snapshot, skill);
    const maxed = rank >= skill.maxRank;
    const poolId = getSkillPointPoolId(snapshot, skill);
    const poolPoints = Number(snapshot.state.player[poolId] || 0);
    const passive = isPassiveSkill(skill);
    const usable = unlocked && rank > 0 && !passive;
    const trainable = unlocked && !maxed && poolPoints > 0;
    const missing = getMissingPrerequisiteLabels(snapshot, skill);
    const stateLabel = !unlocked
      ? 'Locked'
      : maxed
        ? 'Maxed'
        : trainable
          ? 'Trainable'
          : usable
            ? 'Usable'
            : rank > 0
              ? 'Learned'
              : 'Visible';
    return {
      rank,
      unlocked,
      maxed,
      poolId,
      poolPoints,
      passive,
      usable,
      trainable,
      missing,
      stateLabel
    };
  }

  function getSkillActivationDomAction(target, clickCount) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const skillUseId = getAttribute('data-starfall-skill-use');
    if (skillUseId && Number(clickCount || 0) >= 2) return { handled: true, type: 'activateSkillUse', skillId: skillUseId };
    const useSkillId = getAttribute('data-starfall-use-skill');
    if (useSkillId) return { handled: true, type: 'activateSkill', skillId: useSkillId };
    return { handled: false, type: '' };
  }

  function getCanvasSkillActivationRegion(region, options) {
    if (!region) return null;
    if (region.type === 'skill-card' && region.usable && region.skillId) {
      return { type: 'skill', id: String(region.skillId) };
    }
    if (region.type === 'bind-action') {
      const settings = options || {};
      const skillBindPrefix = settings.skillBindPrefix || 'skill:';
      const actionId = String(region.actionId || '');
      if (region.skillId) return { type: 'skill', id: String(region.skillId) };
      if (skillBindPrefix && actionId.startsWith(skillBindPrefix)) {
        return { type: 'skill', id: actionId.slice(skillBindPrefix.length) };
      }
    }
    return null;
  }

  function getCanvasSkillActivationKey(activation) {
    return activation ? `${activation.type}:${activation.id}` : '';
  }

  function isCanvasSkillDoubleClick(activation, point, lastClick, now, options) {
    const last = lastClick || null;
    const currentPoint = point || null;
    if (!last || !activation || !currentPoint) return false;
    if (last.key !== getCanvasSkillActivationKey(activation)) return false;
    const settings = options || {};
    const doubleClickMs = Object.prototype.hasOwnProperty.call(settings, 'doubleClickMs')
      ? Number(settings.doubleClickMs)
      : 360;
    const doubleClickDistance = Object.prototype.hasOwnProperty.call(settings, 'doubleClickDistance')
      ? Number(settings.doubleClickDistance)
      : 8;
    const maxMs = Number.isFinite(doubleClickMs) ? doubleClickMs : 360;
    const maxDistance = Number.isFinite(doubleClickDistance) ? doubleClickDistance : 8;
    if (Number(now || 0) - Number(last.time || 0) > maxMs) return false;
    return Math.hypot(Number(currentPoint.x || 0) - Number(last.x || 0), Number(currentPoint.y || 0) - Number(last.y || 0)) <= maxDistance;
  }

  function getCanvasSkillActivationClickAction(activation, point, isDoubleClick, now) {
    if (!activation) {
      return {
        handled: false,
        type: '',
        activation: null,
        skillId: '',
        click: null,
        shouldClearCanvasSkillClick: true,
        shouldClearSelectedBind: false,
        shouldActivateSkill: false,
        shouldRememberCanvasSkillClick: false,
        shouldPreventDefault: false,
        shouldReturnEarly: false
      };
    }
    if (isDoubleClick) {
      return {
        handled: true,
        type: 'activateSkillSelection',
        activation,
        skillId: activation.id,
        click: null,
        shouldClearCanvasSkillClick: true,
        shouldClearSelectedBind: true,
        shouldActivateSkill: true,
        shouldRememberCanvasSkillClick: false,
        shouldPreventDefault: true,
        shouldReturnEarly: true
      };
    }
    const currentPoint = point || {};
    return {
      handled: true,
      type: 'rememberSkillActivationClick',
      activation,
      skillId: activation.id,
      click: {
        key: getCanvasSkillActivationKey(activation),
        x: currentPoint.x,
        y: currentPoint.y,
        time: now
      },
      shouldClearCanvasSkillClick: false,
      shouldClearSelectedBind: false,
      shouldActivateSkill: false,
      shouldRememberCanvasSkillClick: true,
      shouldPreventDefault: false,
      shouldReturnEarly: false
    };
  }

  function getSkillActivationSelectionAction(skillId) {
    const id = String(skillId || '');
    if (!id) return { handled: false, type: '', skillId: '', shouldFocusCanvas: false };
    return { handled: true, type: 'useSkill', skillId: id, shouldFocusCanvas: true };
  }

  function getSkillPanelDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const skillId = getAttribute('data-starfall-skill');
    if (skillId) return { handled: true, type: 'rankSkill', skillId };
    const ownerId = getAttribute('data-starfall-skill-tab');
    if (ownerId) return { handled: true, type: 'selectOwner', ownerId };
    return { handled: false, type: '' };
  }

  function getSkillPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'skill-rank') return { handled: true, type: 'rankSkill', skillId: source.skillId };
    if (source.type === 'skill-tab') return { handled: true, type: 'selectOwner', ownerId: source.owner };
    if (source.type === 'advanced') return { handled: true, type: 'chooseAdvancedClass', advancedId: source.advancedId };
    if (source.type === 'specialization') return { handled: true, type: 'chooseSpecialization', specializationId: source.specializationId };
    return { handled: false, type: '' };
  }

  function getAdvancedClassDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const advancedId = getAttribute('data-starfall-advanced');
    if (advancedId) return { handled: true, type: 'chooseAdvancedClass', advancedId };
    return { handled: false, type: '' };
  }

  function createSkillStateUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      getSkillById: typeof settings.getSkillById === 'function' ? settings.getSkillById : (skillId) => getSkillById(skillId, settings),
      getSkillPointPoolId: typeof settings.getSkillPointPoolId === 'function' ? settings.getSkillPointPoolId : UiSkillMetadata.getSkillPointPoolId,
      isPassiveSkill: typeof settings.isPassiveSkill === 'function' ? settings.isPassiveSkill : UiSkillMetadata.isPassiveSkill || (() => false),
      getMissingPrerequisiteLabels: typeof settings.getMissingPrerequisiteLabels === 'function' ? settings.getMissingPrerequisiteLabels : UiSkillPrerequisites.getMissingPrerequisiteLabels || (() => []),
      skillBindPrefix: settings.skillBindPrefix || 'skill:',
      doubleClickMs: Object.prototype.hasOwnProperty.call(settings, 'doubleClickMs') ? settings.doubleClickMs : 360,
      doubleClickDistance: Object.prototype.hasOwnProperty.call(settings, 'doubleClickDistance') ? settings.doubleClickDistance : 8
    });
    function getOptions(extra) {
      return Object.assign({}, helperOptions, extra || {});
    }
    return Object.freeze({
      isSkillUnlocked,
      describePrerequisites: (skill, extraOptions) => describePrerequisites(skill, getOptions(extraOptions)),
      getSkillUiState: (snapshot, skill, extraOptions) => getSkillUiState(snapshot, skill, getOptions(extraOptions)),
      getSkillActivationDomAction,
      getCanvasSkillActivationRegion: (region, extraOptions) => getCanvasSkillActivationRegion(region, getOptions(extraOptions)),
      getCanvasSkillActivationKey,
      isCanvasSkillDoubleClick: (activation, point, lastClick, now, extraOptions) => isCanvasSkillDoubleClick(activation, point, lastClick, now, getOptions(extraOptions)),
      getCanvasSkillActivationClickAction,
      getSkillActivationSelectionAction,
      getSkillPanelDomAction,
      getSkillPanelRegionAction,
      getAdvancedClassDomAction
    });
  }

  const api = {
    isSkillUnlocked,
    describePrerequisites,
    getSkillUiState,
    getSkillActivationDomAction,
    getCanvasSkillActivationRegion,
    getCanvasSkillActivationKey,
    isCanvasSkillDoubleClick,
    getCanvasSkillActivationClickAction,
    getSkillActivationSelectionAction,
    getSkillPanelDomAction,
    getSkillPanelRegionAction,
    getAdvancedClassDomAction,
    createSkillStateUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.skillState = Object.assign({}, modules.skillState || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
