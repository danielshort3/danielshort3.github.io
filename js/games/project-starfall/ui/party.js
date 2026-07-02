(function initProjectStarfallUiParty(global) {
  'use strict';

  function getPartyPanelDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (hasAttribute('data-starfall-party-find')) return { handled: true, type: 'find' };
    if (hasAttribute('data-starfall-party-reroll')) return { handled: true, type: 'reroll' };
    if (hasAttribute('data-starfall-party-clear')) return { handled: true, type: 'clear' };
    const commandId = getAttribute('data-starfall-party-command');
    if (commandId) return { handled: true, type: 'setCommand', commandId };
    return { handled: false, type: '' };
  }

  function getPartyPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'party-find') return { handled: true, type: 'find' };
    if (source.type === 'party-reroll') return { handled: true, type: 'reroll' };
    if (source.type === 'party-clear') return { handled: true, type: 'clear' };
    if (source.type === 'party-command') return { handled: true, type: 'setCommand', commandId: source.commandId };
    if (source.type === 'party-skill') return { handled: true, type: 'openPartyPanel' };
    return { handled: false, type: '' };
  }

  function createPartyPanelUiHelpers() {
    return Object.freeze({
      getPartyPanelDomAction,
      getPartyPanelRegionAction
    });
  }

  const api = {
    createPartyPanelUiHelpers,
    getPartyPanelDomAction,
    getPartyPanelRegionAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.party = Object.assign({}, modules.party || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
