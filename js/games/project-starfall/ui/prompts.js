(function initProjectStarfallUiPrompts(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiCanvasWindows = (typeof require === 'function' ? require('./canvas-windows.js') : null) || UiModules.canvasWindows || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  function normalizeConfirmPromptQuantity(quantity) {
    if (!quantity || typeof quantity !== 'object') return null;
    const min = Math.max(1, Math.floor(Number(quantity.min || 1) || 1));
    const max = Math.max(1, Math.floor(Number(quantity.max || 1) || 1));
    return {
      value: clamp(Math.floor(Number(quantity.value || quantity.max || 1) || 1), min, max),
      min,
      max,
      label: quantity.label || 'Quantity',
      unit: quantity.unit || ''
    };
  }

  function createConfirmPrompt(config) {
    const settings = config || {};
    if (typeof settings.onConfirm !== 'function') return null;
    return {
      id: settings.id || 'confirm',
      title: settings.title || 'Confirm',
      message: settings.message || 'Continue?',
      confirmLabel: settings.confirmLabel || 'Confirm',
      cancelLabel: settings.cancelLabel || 'Cancel',
      tone: settings.tone || '',
      quantity: normalizeConfirmPromptQuantity(settings.quantity),
      messageBuilder: typeof settings.messageBuilder === 'function' ? settings.messageBuilder : null,
      onConfirm: settings.onConfirm
    };
  }

  function getConfirmPromptMessage(prompt) {
    const source = prompt || {};
    if (source.messageBuilder && source.quantity) {
      return source.messageBuilder(source.quantity.value, source) || source.message || 'Continue?';
    }
    return source.message || 'Continue?';
  }

  function normalizeConfirmPromptQuantityValue(value, quantity) {
    const source = quantity || {};
    return clamp(Math.floor(Number(value || source.value) || source.value), source.min, source.max);
  }

  function getAdjustedConfirmPromptQuantityValue(prompt, delta) {
    const quantity = prompt && prompt.quantity || null;
    if (!quantity) return null;
    return normalizeConfirmPromptQuantityValue(Number(quantity.value || 1) + Number(delta || 0), quantity);
  }

  function getConfirmPromptDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (hasAttribute('data-starfall-confirm-cancel')) return { handled: true, type: 'cancel' };
    if (hasAttribute('data-starfall-confirm-accept')) return { handled: true, type: 'confirm' };
    const delta = getAttribute('data-starfall-confirm-quantity-delta');
    if (delta) return { handled: true, type: 'adjustQuantity', delta };
    return { handled: false, type: '' };
  }

  function getConfirmPromptInputDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (!hasAttribute('data-starfall-confirm-quantity-input')) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'setQuantity',
      value: source && source.value
    };
  }

  function getForegroundDomPromptSelector(state) {
    const source = state || {};
    if (source.confirmPrompt) {
      return {
        handled: true,
        type: 'confirm',
        selector: '.project-starfall-confirm-prompt'
      };
    }
    if (source.potentialPromptOpen) {
      return {
        handled: true,
        type: 'potential',
        selector: '.project-starfall-potential-confirm'
      };
    }
    if (source.shardCraftPromptOpen) {
      return {
        handled: true,
        type: 'shardCraft',
        selector: '.project-starfall-shard-craft'
      };
    }
    if (source.upgradePromptOpen) {
      return {
        handled: true,
        type: 'upgrade',
        selector: '.project-starfall-upgrade-confirm'
      };
    }
    return { handled: false, type: '', selector: '' };
  }

  function getForegroundPromptClickGuardAction(state) {
    const source = state || {};
    const shouldBlockClick = !!(source.hasForegroundPrompt && source.hasTarget && !source.targetInsidePrompt);
    return {
      handled: shouldBlockClick,
      type: shouldBlockClick ? 'blockOutsideForegroundPromptClick' : '',
      shouldBlockClick
    };
  }

  function getPromptRegionAction(region) {
    const source = region || {};
    if (source.type === 'drop-quantity-decrease') {
      return { handled: true, group: 'dropQuantity', type: 'adjustDropQuantity', delta: -1 };
    }
    if (source.type === 'drop-quantity-increase') {
      return { handled: true, group: 'dropQuantity', type: 'adjustDropQuantity', delta: 1 };
    }
    if (source.type === 'drop-quantity-all') {
      return { handled: true, group: 'dropQuantity', type: 'setDropQuantityMax' };
    }
    if (source.type === 'drop-quantity-cancel') {
      return { handled: true, group: 'dropQuantity', type: 'cancelDropQuantity' };
    }
    if (source.type === 'drop-quantity-confirm') {
      return { handled: true, group: 'dropQuantity', type: 'confirmDropQuantity' };
    }
    if (source.type === 'confirm-prompt-cancel') {
      return { handled: true, group: 'confirm', type: 'cancelConfirmPrompt' };
    }
    if (source.type === 'confirm-prompt-quantity-decrease') {
      return { handled: true, group: 'confirm', type: 'adjustConfirmQuantity', delta: -1 };
    }
    if (source.type === 'confirm-prompt-quantity-increase') {
      return { handled: true, group: 'confirm', type: 'adjustConfirmQuantity', delta: 1 };
    }
    if (source.type === 'confirm-prompt-accept') {
      return { handled: true, group: 'confirm', type: 'confirmPrompt' };
    }
    return { handled: false, group: '', type: '' };
  }

  function getGearPickerRegionAction(region) {
    const source = region || {};
    if (source.type === 'gear-picker-open') {
      return { handled: true, type: 'open', context: source.context };
    }
    if (source.type === 'gear-picker-close') return { handled: true, type: 'close' };
    if (source.type === 'gear-picker-item') {
      return { handled: true, type: 'selectItem', uid: source.uid };
    }
    return { handled: false, type: '' };
  }

  function getPromptPointerAction(region) {
    const source = region || {};
    if (source.type === 'quest-prompt-header') {
      return {
        handled: true,
        type: 'startPromptDrag',
        dragKey: 'questPromptDrag',
        boxType: 'questPrompt',
        activePanel: 'questPrompt',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'drop-quantity-header') {
      return {
        handled: true,
        type: 'startPromptDrag',
        dragKey: 'dropQuantityPromptDrag',
        boxType: 'dropQuantityPrompt',
        activePanel: 'dropQuantityPrompt',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'admin-number-header') {
      return {
        handled: true,
        type: 'startPromptDrag',
        dragKey: 'adminNumberPromptDrag',
        boxType: 'adminNumberPrompt',
        activePanel: 'adminNumberPrompt',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'confirm-prompt-header') {
      return {
        handled: true,
        type: 'startPromptDrag',
        dragKey: 'confirmPromptDrag',
        boxType: 'confirmPrompt',
        activePanel: 'confirmPrompt',
        shouldPreventDefault: true
      };
    }
    return {
      handled: false,
      type: '',
      dragKey: '',
      boxType: '',
      activePanel: '',
      shouldPreventDefault: false
    };
  }

  function getPromptBox(width, height, state, options) {
    const settings = options || {};
    const getCenteredPromptBox = typeof settings.getCenteredPromptBox === 'function'
      ? settings.getCenteredPromptBox
      : UiCanvasWindows.getCenteredPromptBox;
    if (getCenteredPromptBox) {
      const centeredOptions = Object.assign({}, settings);
      delete centeredOptions.getCenteredPromptBox;
      return getCenteredPromptBox(width, height, state, centeredOptions);
    }
    const horizontalInset = Object.prototype.hasOwnProperty.call(settings, 'horizontalInset')
      ? Number(settings.horizontalInset)
      : 32;
    const minWidth = Number(settings.minWidth || 286);
    const maxWidth = Number(settings.maxWidth || 360);
    const boxW = Math.min(maxWidth, Math.max(minWidth, width - horizontalInset));
    const boxH = Number(settings.height || 0);
    const bottomLimit = Number(settings.bottomLimit || height);
    const defaultYMin = Object.prototype.hasOwnProperty.call(settings, 'defaultYMin')
      ? Number(settings.defaultYMin)
      : 16;
    const defaultBottomInset = Object.prototype.hasOwnProperty.call(settings, 'defaultBottomInset')
      ? Number(settings.defaultBottomInset)
      : 8;
    const defaultYOffset = Object.prototype.hasOwnProperty.call(settings, 'defaultYOffset')
      ? Number(settings.defaultYOffset)
      : 0;
    const shouldConstrainDefaultY = settings.constrainDefaultY !== false;
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, userPlaced: false };
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      const centeredY = (bottomLimit - boxH) / 2 + defaultYOffset;
      target.y = Math.round(Math.max(defaultYMin, shouldConstrainDefaultY
        ? Math.min(bottomLimit - boxH - defaultBottomInset, centeredY)
        : centeredY));
    }
    target.x = clamp(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clamp(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getQuestPromptBox(width, height, state, options) {
    const settings = options || {};
    return getPromptBox(width, height, state, {
      getCenteredPromptBox: settings.getCenteredPromptBox,
      minWidth: 300,
      maxWidth: 460,
      height: 286,
      bottomLimit: settings.bottomLimit,
      defaultYMin: 18,
      constrainDefaultY: false
    });
  }

  function getDropQuantityPromptBox(width, height, state, options) {
    const settings = options || {};
    return getPromptBox(width, height, state, {
      getCenteredPromptBox: settings.getCenteredPromptBox,
      minWidth: 286,
      maxWidth: 360,
      height: 176,
      bottomLimit: settings.bottomLimit
    });
  }

  function getAdminNumberPromptBox(width, height, state, options) {
    const settings = options || {};
    return getPromptBox(width, height, state, {
      getCenteredPromptBox: settings.getCenteredPromptBox,
      minWidth: 286,
      maxWidth: 340,
      height: 214,
      bottomLimit: settings.bottomLimit
    });
  }

  function getConfirmPromptBox(width, height, state, options) {
    const settings = options || {};
    return getPromptBox(width, height, state, {
      getCenteredPromptBox: settings.getCenteredPromptBox,
      minWidth: 286,
      maxWidth: 360,
      height: settings.hasQuantity ? 222 : 188,
      bottomLimit: settings.bottomLimit
    });
  }

  function getGearPickerBox(width, height, state, options) {
    const settings = options || {};
    const bottomLimit = Number(settings.bottomLimit || height);
    const boxW = Math.min(440, Math.max(320, width - 32));
    const boxH = Math.min(430, Math.max(240, bottomLimit - 24));
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, scroll: 0 };
    target.w = boxW;
    target.h = boxH;
    target.x = Math.round((width - boxW) / 2);
    target.y = Math.round(Math.max(12, (bottomLimit - boxH) / 2));
    target.scroll = clamp(Number(target.scroll || 0), 0, Number(target.maxScroll || 0));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function createPromptUiHelpers() {
    return Object.freeze({
      normalizeConfirmPromptQuantity,
      createConfirmPrompt,
      getConfirmPromptMessage,
      normalizeConfirmPromptQuantityValue,
      getAdjustedConfirmPromptQuantityValue,
      getConfirmPromptDomAction,
      getConfirmPromptInputDomAction,
      getForegroundDomPromptSelector,
      getForegroundPromptClickGuardAction,
      getPromptRegionAction,
      getGearPickerRegionAction,
      getPromptPointerAction,
      getPromptBox,
      getQuestPromptBox,
      getDropQuantityPromptBox,
      getAdminNumberPromptBox,
      getConfirmPromptBox,
      getGearPickerBox
    });
  }

  const api = {
    createPromptUiHelpers,
    normalizeConfirmPromptQuantity,
    createConfirmPrompt,
    getConfirmPromptMessage,
    normalizeConfirmPromptQuantityValue,
    getAdjustedConfirmPromptQuantityValue,
    getConfirmPromptDomAction,
    getConfirmPromptInputDomAction,
    getForegroundDomPromptSelector,
    getForegroundPromptClickGuardAction,
    getPromptRegionAction,
    getGearPickerRegionAction,
    getPromptPointerAction,
    getPromptBox,
    getQuestPromptBox,
    getDropQuantityPromptBox,
    getAdminNumberPromptBox,
    getConfirmPromptBox,
    getGearPickerBox
  };

  const modules = UiModules;
  modules.prompts = Object.assign({}, modules.prompts || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
