(function initProjectStarfallUiInput(global) {
  'use strict';

  const DOM_CLICK_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-confirm-accept',
    'data-starfall-confirm-cancel',
    'data-starfall-confirm-quantity-delta',
    'data-starfall-action',
    'data-starfall-open-panel',
    'data-starfall-command-toggle',
    'data-starfall-command-channel',
    'data-starfall-close',
    'data-starfall-rebind',
    'data-starfall-bind-action',
    'data-starfall-key-target',
    'data-starfall-disabled-key',
    'data-starfall-clear-key',
    'data-starfall-reset-keybinds',
    'data-starfall-reset-settings',
    'data-starfall-setting-preset',
    'data-starfall-setting-hud',
    'data-starfall-setting-damage',
    'data-starfall-reset-admin',
    'data-starfall-admin-console-restore',
    'data-starfall-admin-console-open',
    'data-starfall-admin-console-close',
    'data-starfall-admin-console-tab',
    'data-starfall-admin-command-run',
    'data-starfall-admin-command-sample',
    'data-starfall-admin-command-clear',
    'data-starfall-admin-picker-toggle',
    'data-starfall-admin-picker-option',
    'data-starfall-admin-number-open',
    'data-starfall-admin-number-delta',
    'data-starfall-admin-number-min',
    'data-starfall-admin-number-max',
    'data-starfall-admin-number-clear',
    'data-starfall-admin-number-cancel',
    'data-starfall-admin-number-apply',
    'data-starfall-admin-spawn',
    'data-starfall-admin-kill-enemies',
    'data-starfall-admin-boss-teleport',
    'data-starfall-admin-grant',
    'data-starfall-admin-gear-min',
    'data-starfall-admin-gear-max',
    'data-starfall-admin-gear-apply',
    'data-starfall-admin-attunement-apply',
    'data-starfall-asset-preview-open',
    'data-starfall-asset-preview-category',
    'data-starfall-asset-preview-source',
    'data-starfall-asset-preview-select',
    'data-starfall-asset-preview-state',
    'data-starfall-asset-preview-toggle',
    'data-starfall-performance-debug-mode',
    'data-starfall-copy-performance-debug',
    'data-starfall-performance-benchmark',
    'data-starfall-copy-performance-benchmark',
    'data-starfall-copy-admin-debug',
    'data-starfall-clear-admin-debug',
    'data-starfall-toggle-combat-metrics',
    'data-starfall-dismiss-guide',
    'data-starfall-character-page',
    'data-starfall-character-slot',
    'data-starfall-character-start',
    'data-starfall-character-create-open',
    'data-starfall-character-create-confirm',
    'data-starfall-character-create-cancel',
    'data-starfall-character-delete',
    'data-starfall-character-delete-cancel',
    'data-starfall-character-name',
    'data-starfall-character-popover',
    'data-starfall-character-look',
    'data-starfall-character-class',
    'data-starfall-character-tab',
    'data-starfall-stat-upgrade',
    'data-starfall-stat-reset',
    'data-starfall-party-find',
    'data-starfall-party-reroll',
    'data-starfall-party-clear',
    'data-starfall-party-command',
    'data-starfall-target-farm',
    'data-starfall-world-map-node',
    'data-starfall-world-map-guide',
    'data-starfall-equipment-tab',
    'data-starfall-card-deck',
    'data-starfall-card-equip',
    'data-starfall-card-unequip',
    'data-starfall-card-upgrade',
    'data-starfall-card-upgrade-all',
    'data-starfall-card-sell',
    'data-starfall-card-lock',
    'data-starfall-star-card-exchange',
    'data-starfall-select-card',
    'data-starfall-apply-inventory-sort',
    'data-starfall-inventory-sort',
    'data-starfall-inventory-tab',
    'data-starfall-storage-tab',
    'data-starfall-inventory-locked',
    'data-starfall-inventory-section-unlock',
    'data-starfall-inventory-slot-purchase',
    'data-starfall-inventory-sell-settings',
    'data-starfall-inventory-sell-toggle',
    'data-starfall-inventory-sell-rarity',
    'data-starfall-inventory-sell-reset',
    'data-starfall-pet-toggle',
    'data-starfall-pet-threshold',
    'data-starfall-pet-picker',
    'data-starfall-pet-pick',
    'data-starfall-pet-loot-filter',
    'data-starfall-toggle-lock',
    'data-starfall-bulk-sell',
    'data-starfall-class',
    'data-starfall-skill',
    'data-starfall-skill-use',
    'data-starfall-skill-tab',
    'data-starfall-shop-buy-entry',
    'data-starfall-shop-sell-item',
    'data-starfall-shop-sell-bulk',
    'data-starfall-buy',
    'data-starfall-cash-shop-buy',
    'data-starfall-daily-login-claim',
    'data-starfall-plinko-select',
    'data-starfall-plinko-buy-quantity',
    'data-starfall-plinko-buy',
    'data-starfall-plinko-drop',
    'data-starfall-plinko-claim-all',
    'data-starfall-equip',
    'data-starfall-unequip',
    'data-starfall-select-item',
    'data-starfall-select-consumable',
    'data-starfall-select-etc',
    'data-starfall-select-upgrade',
    'data-starfall-upgrade-drop-zone',
    'data-starfall-upgrade-aide',
    'data-starfall-upgrade-confirm',
    'data-starfall-upgrade-close',
    'data-starfall-potential-drop-zone',
    'data-starfall-potential-confirm',
    'data-starfall-potential-preserve',
    'data-starfall-potential-repeat',
    'data-starfall-potential-line-upgrade',
    'data-starfall-potential-choice',
    'data-starfall-potential-auto-toggle',
    'data-starfall-potential-auto-close',
    'data-starfall-potential-auto-stat-toggle',
    'data-starfall-potential-auto-stat-add',
    'data-starfall-potential-auto-stat-remove',
    'data-starfall-potential-auto-repeat',
    'data-starfall-potential-auto-run',
    'data-starfall-potential-help',
    'data-starfall-potential-close',
    'data-starfall-shard-craft-close',
    'data-starfall-shard-craft-recipe',
    'data-starfall-shard-craft-delta',
    'data-starfall-shard-craft-all',
    'data-starfall-shard-craft-confirm',
    'data-starfall-combine-cube-fragments',
    'data-starfall-combine-preservation-cube-fragments',
    'data-starfall-use-skill',
    'data-starfall-use-consumable',
    'data-starfall-advanced'
  ]);
  const DOM_CLICK_TARGET_SELECTOR = DOM_CLICK_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_INPUT_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-admin-rate',
    'data-starfall-admin-console-control',
    'data-starfall-admin-command-input',
    'data-starfall-asset-preview-category',
    'data-starfall-asset-preview-source',
    'data-starfall-asset-preview-query',
    'data-starfall-setting',
    'data-starfall-pet-potion',
    'data-starfall-pet-min-rarity',
    'data-starfall-inventory-sort-choice',
    'data-starfall-character-name',
    'data-starfall-confirm-quantity-input',
    'data-starfall-potential-auto-stat',
    'data-starfall-potential-auto-stat-row',
    'data-starfall-potential-auto-stat-min',
    'data-starfall-potential-auto-stat-min-index',
    'data-starfall-potential-auto-tier',
    'data-starfall-potential-auto-max'
  ]);
  const DOM_INPUT_TARGET_SELECTOR = DOM_INPUT_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_SKILL_TAB_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-skill-tab'
  ]);
  const DOM_SKILL_TAB_TARGET_SELECTOR = DOM_SKILL_TAB_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_INVENTORY_TAB_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-inventory-tab'
  ]);
  const DOM_INVENTORY_TAB_TARGET_SELECTOR = DOM_INVENTORY_TAB_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_CHARACTER_TAB_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-character-tab'
  ]);
  const DOM_CHARACTER_TAB_TARGET_SELECTOR = DOM_CHARACTER_TAB_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_COMMAND_PANEL_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-command-panel'
  ]);
  const DOM_COMMAND_PANEL_TARGET_SELECTOR = DOM_COMMAND_PANEL_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_TOUCH_CONTROL_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-touch-action'
  ]);
  const DOM_TOUCH_CONTROL_TARGET_SELECTOR = DOM_TOUCH_CONTROL_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function copySet(value) {
    if (value && typeof value.forEach === 'function') {
      const next = new Set();
      value.forEach((entry) => {
        next.add(entry);
      });
      return next;
    }
    if (Array.isArray(value)) return new Set(value);
    return new Set();
  }

  function copyMap(value) {
    if (value && typeof value.forEach === 'function' && typeof value.set === 'function') {
      const next = new Map();
      value.forEach((entryValue, entryKey) => {
        next.set(entryKey, entryValue);
      });
      return next;
    }
    if (Array.isArray(value)) return new Map(value);
    return new Map();
  }

  function getAttackKeyInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const keyCode = String(code || '');
    const nextHeldKeys = copySet(currentState.heldAttackKeys);
    const wasHolding = nextHeldKeys.size > 0;
    if (isDown) nextHeldKeys.add(keyCode);
    else nextHeldKeys.delete(keyCode);
    const holding = nextHeldKeys.size > 0;
    return {
      handled: true,
      keyCode,
      isDown: !!isDown,
      repeat: !!currentState.repeat,
      wasHolding,
      holding,
      heldAttackKeys: nextHeldKeys,
      shouldSetAttackInput: true,
      attackInput: holding,
      shouldBasicAttack: !!isDown && !wasHolding && !currentState.repeat,
      basicAttackOptions: { silent: true, fromHeldInput: true }
    };
  }

  function getAttackKeyInputStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldSetHeldAttackKeys: !!currentMetadata.handled,
      heldAttackKeys: currentMetadata.heldAttackKeys instanceof Set
        ? currentMetadata.heldAttackKeys
        : copySet(currentMetadata.heldAttackKeys)
    };
  }

  function getSkillKeyInputMetadata(action, code, isDown, state) {
    const currentAction = action || {};
    const skillId = String(currentAction.skillId || '');
    if (!skillId) {
      return {
        handled: false,
        keyCode: String(code || ''),
        skillId: '',
        heldSkillKeys: copyMap(state && state.heldSkillKeys)
      };
    }
    const currentState = state || {};
    const keyCode = String(code || '');
    const nextHeldKeys = copyMap(currentState.heldSkillKeys);
    if (isDown) {
      nextHeldKeys.set(keyCode, skillId);
      return {
        handled: true,
        mode: 'press',
        keyCode,
        skillId,
        repeat: !!currentState.repeat,
        heldSkillKeys: nextHeldKeys,
        shouldSetEngineHeldSkill: true,
        engineHeldSkillValue: true,
        shouldUseSkill: !currentState.repeat
      };
    }
    nextHeldKeys.delete(keyCode);
    const stillHeld = Array.from(nextHeldKeys.values()).includes(skillId);
    return {
      handled: true,
      mode: 'release',
      keyCode,
      skillId,
      repeat: !!currentState.repeat,
      heldSkillKeys: nextHeldKeys,
      shouldSetEngineHeldSkill: !stillHeld,
      engineHeldSkillValue: false,
      shouldUseSkill: false
    };
  }

  function getSkillKeyInputStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldSetHeldSkillKeys: !!currentMetadata.handled,
      heldSkillKeys: currentMetadata.heldSkillKeys instanceof Map
        ? currentMetadata.heldSkillKeys
        : copyMap(currentMetadata.heldSkillKeys)
    };
  }

  function getClearHoldInputsMetadata(inputNames) {
    const sourceNames = Array.isArray(inputNames) && inputNames.length
      ? inputNames
      : ['left', 'right', 'up', 'jump', 'down', 'attack', 'loot'];
    return {
      shouldClearHeldAttackKeys: true,
      shouldClearHeldSkillKeys: true,
      shouldClearEngineHeldSkills: true,
      inputStates: sourceNames.map((name) => ({
        name: String(name || ''),
        value: false
      })).filter((entry) => entry.name)
    };
  }

  function getClearHoldInputsStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearHeldAttackKeys: !!currentMetadata.shouldClearHeldAttackKeys,
      shouldClearHeldSkillKeys: !!currentMetadata.shouldClearHeldSkillKeys
    };
  }

  function getReleaseAttackInputMetadata() {
    return {
      shouldClearHeldAttackKeys: true,
      shouldSetAttackInput: true,
      attackInput: false
    };
  }

  function getReleaseAttackInputStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearHeldAttackKeys: !!currentMetadata.shouldClearHeldAttackKeys
    };
  }

  function getDomPointerReleaseCleanupMetadata(options) {
    const settings = options || {};
    return {
      shouldCommitAdminRatePreview: !!settings.shouldCommitAdminRatePreview,
      shouldStopPlinkoDropHold: true,
      plinkoStopOptions: settings.keepConsumedClick ? { keepConsumedClick: true } : null,
      shouldReleaseAttackInput: !settings.preserveTouchAttack,
      shouldStopPotentialPromptDomDrag: true
    };
  }

  function getDomAttackButtonPointerAction(pointerAction, options) {
    const source = pointerAction || {};
    const settings = options || {};
    const handled = !!(source.handled && settings.inRoot && !source.disabled);
    return {
      handled,
      type: handled ? 'pressAttack' : '',
      actionId: source.actionId || 'attack',
      target: source.target || null,
      disabled: !!source.disabled,
      inRoot: !!settings.inRoot,
      shouldSetAttackInput: handled,
      attackInput: true,
      shouldBasicAttack: handled,
      shouldFocusCanvas: handled,
      shouldPreventDefault: handled
    };
  }

  function getTouchControlTarget(target, selector) {
    const source = target || null;
    const touchSelector = selector || DOM_TOUCH_CONTROL_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(touchSelector)
      : null;
  }

  function getTouchControlPointerAction(event, options) {
    const source = event || {};
    const settings = options || {};
    const target = getTouchControlTarget(source.target, settings.selector);
    const pointerId = source.pointerId == null ? '' : String(source.pointerId);
    const pointerType = String(source.pointerType || 'touch');
    const actionId = target && typeof target.getAttribute === 'function'
      ? String(target.getAttribute('data-starfall-touch-action') || '')
      : '';
    const inRoot = !!(target && settings.root && typeof settings.root.contains === 'function' && settings.root.contains(target));
    const disabled = !!(target && (target.disabled || target.getAttribute && target.getAttribute('aria-disabled') === 'true'));
    const unsupportedMouseButton = pointerType === 'mouse' && Number(source.button || 0) !== 0;
    const handled = !!(target && pointerId && actionId && inRoot && !disabled && !unsupportedMouseButton);
    return {
      handled,
      target,
      actionId,
      pointerId,
      pointerKey: handled ? `touch:${pointerId}` : '',
      pointerType,
      disabled,
      inRoot,
      shouldSetPointerCapture: handled && Number.isFinite(Number(source.pointerId)) && typeof target.setPointerCapture === 'function',
      shouldPreventDefault: handled,
      shouldFocusCanvas: handled
    };
  }

  function getTouchControlReleaseAction(event, activePointers) {
    const source = event || {};
    const pointers = activePointers && typeof activePointers.get === 'function' ? activePointers : new Map();
    const pointerId = source.pointerId == null ? '' : String(source.pointerId);
    const pointer = pointerId ? pointers.get(pointerId) || null : null;
    return {
      handled: !!pointer,
      pointerId,
      pointer,
      actionId: pointer && pointer.actionId || '',
      pointerKey: pointer && pointer.pointerKey || (pointerId ? `touch:${pointerId}` : ''),
      target: pointer && pointer.target || null,
      shouldReleasePointerCapture: !!(
        pointer &&
        pointer.target &&
        Number.isFinite(Number(source.pointerId)) &&
        typeof pointer.target.releasePointerCapture === 'function'
      )
    };
  }

  function getOutOfRootClickAction(state) {
    const source = state || {};
    const outsideRoot = !source.hasTarget || !source.targetInRoot;
    if (!outsideRoot) {
      return {
        handled: false,
        type: '',
        shouldCloseCharacterSelectPopover: false,
        shouldRenderClassSelect: false,
        shouldClearSelectedBindFromOutside: false
      };
    }
    const shouldCloseCharacterSelectPopover = !!(
      source.isCharacterSelectOpen &&
      source.characterSelectPopoverActive &&
      !source.characterCreateDraftActive
    );
    return {
      handled: true,
      type: shouldCloseCharacterSelectPopover ? 'closeCharacterSelectPopover' : 'clearSelectedBindFromOutside',
      shouldCloseCharacterSelectPopover,
      shouldRenderClassSelect: shouldCloseCharacterSelectPopover,
      shouldClearSelectedBindFromOutside: !shouldCloseCharacterSelectPopover
    };
  }

  function getDomClickTarget(target, selector) {
    const source = target || null;
    const clickSelector = selector || DOM_CLICK_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(clickSelector)
      : null;
  }

  function getDomInputTarget(target, selector) {
    const source = target || null;
    const inputSelector = selector || DOM_INPUT_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(inputSelector)
      : null;
  }

  function getFixedMovementInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const movementKeys = currentState.fixedMovementKeys || {};
    const keyCode = String(code || '');
    const input = String(movementKeys[keyCode] || '');
    if (!input) {
      return {
        handled: false,
        keyCode,
        input: '',
        isDown: !!isDown,
        shouldPreventDefault: false,
        shouldTryEnterActivePortal: false,
        shouldSetInput: false,
        inputValue: !!isDown
      };
    }
    return {
      handled: true,
      keyCode,
      input,
      isDown: !!isDown,
      repeat: !!currentState.repeat,
      shouldPreventDefault: true,
      shouldTryEnterActivePortal: input === 'up' && !!isDown && !currentState.repeat,
      shouldSetInput: true,
      inputValue: !!isDown
    };
  }

  function getBoundActionInputDispatch(action, isDown, state) {
    if (!action) {
      return {
        handled: false,
        mode: 'none',
        action: null,
        shouldTriggerAction: false
      };
    }
    const currentState = state || {};
    const down = !!isDown;
    if (action.type === 'hold') {
      return {
        handled: true,
        mode: 'hold',
        action,
        input: action.input,
        inputValue: down,
        shouldTriggerAction: false
      };
    }
    if (action.action === 'attack') {
      return {
        handled: true,
        mode: 'attack',
        action,
        shouldTriggerAction: false
      };
    }
    if (action.type === 'skill') {
      return {
        handled: true,
        mode: 'skill',
        action,
        shouldTriggerAction: false
      };
    }
    return {
      handled: true,
      mode: down && !currentState.repeat ? 'trigger' : 'none',
      action,
      shouldTriggerAction: down && !currentState.repeat
    };
  }

  function getBoundActionsInputMetadata(code, actions) {
    const actionList = Array.isArray(actions) ? actions.slice() : [];
    const handled = actionList.length > 0;
    return {
      handled,
      keyCode: String(code || ''),
      actions: actionList,
      shouldPreventDefault: handled
    };
  }

  function getGameKeyTargetMetadata(target, options) {
    const settings = options || {};
    const ignoredSelector = settings.ignoredSelector || 'a, button, input, textarea, select, [contenteditable="true"]';
    if (!target || !target.closest) {
      return {
        shouldIgnore: false,
        inCommandPanel: false,
        matchesIgnoredTarget: false
      };
    }
    const inCommandPanel = !!getCommandPanelTarget(target, settings.commandPanelSelector);
    const matchesIgnoredTarget = !!target.closest(ignoredSelector);
    return {
      shouldIgnore: !inCommandPanel && matchesIgnoredTarget,
      inCommandPanel,
      matchesIgnoredTarget
    };
  }

  function getCommandPanelTarget(target, selector) {
    const source = target || null;
    const commandPanelSelector = selector || DOM_COMMAND_PANEL_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(commandPanelSelector)
      : null;
  }

  function isModalKeybindAction(action, allowedActionNames) {
    if (!action) return false;
    if (action.type === 'panel') return true;
    return allowedActionNames.indexOf(action.action) !== -1;
  }

  function getModalKeyboardInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const keyCode = String(code || '');
    const repeat = !!currentState.repeat;
    const base = {
      handled: false,
      mode: 'none',
      keyCode,
      repeat,
      needsActions: false,
      shouldPreventDefault: false,
      shouldClosePanel: false,
      shouldFocusAdjacentControl: false,
      focusDirection: 0,
      shouldTriggerActions: false,
      actions: []
    };
    if (!isDown) return base;
    if (keyCode === 'Escape') {
      return Object.assign({}, base, {
        handled: true,
        mode: 'close',
        shouldPreventDefault: true,
        shouldClosePanel: true
      });
    }
    if (keyCode === 'ArrowLeft' || keyCode === 'ArrowRight' || keyCode === 'ArrowUp' || keyCode === 'ArrowDown') {
      return Object.assign({}, base, {
        handled: true,
        mode: 'focus',
        shouldPreventDefault: true,
        shouldFocusAdjacentControl: true,
        focusDirection: keyCode === 'ArrowLeft' || keyCode === 'ArrowUp' ? -1 : 1
      });
    }
    if (!Object.prototype.hasOwnProperty.call(currentState, 'actions')) {
      return Object.assign({}, base, {
        needsActions: true
      });
    }
    const allowedActionNames = Array.isArray(currentState.allowedActionNames)
      ? currentState.allowedActionNames
      : ['save', 'load', 'fullscreen'];
    const actions = Array.isArray(currentState.actions)
      ? currentState.actions.filter((action) => isModalKeybindAction(action, allowedActionNames))
      : [];
    if (!actions.length) return base;
    return Object.assign({}, base, {
      handled: true,
      mode: 'actions',
      shouldPreventDefault: true,
      shouldTriggerActions: !repeat,
      actions
    });
  }

  function getAdjacentModalControlFocusMetadata(controls, activeElement, direction) {
    const controlList = Array.isArray(controls) ? controls : [];
    const focusDirection = Number(direction || 0) || 0;
    if (!controlList.length) {
      return {
        handled: false,
        focusDirection,
        currentIndex: -1,
        nextIndex: -1,
        control: null
      };
    }
    const currentIndex = controlList.indexOf(activeElement || null);
    const nextIndex = currentIndex < 0
      ? 0
      : (currentIndex + focusDirection + controlList.length) % controlList.length;
    return {
      handled: true,
      focusDirection,
      currentIndex,
      nextIndex,
      control: controlList[nextIndex] || null
    };
  }

  function getPanelTabCycleTargetKind(target, selectors) {
    const source = target || null;
    const options = selectors || {};
    if (!source || typeof source.closest !== 'function') return '';
    if (source.closest(options.skillSelector || DOM_SKILL_TAB_TARGET_SELECTOR)) return 'skill';
    if (source.closest(options.inventorySelector || DOM_INVENTORY_TAB_TARGET_SELECTOR)) return 'inventory';
    if (source.closest(options.characterSelector || DOM_CHARACTER_TAB_TARGET_SELECTOR)) return 'character';
    return '';
  }

  function getPanelTabCycleInputMetadata(event, isDown, state) {
    const currentState = state || {};
    const keyCode = event && event.code || '';
    const key = event && event.key || '';
    const delta = event && event.shiftKey ? -1 : 1;
    const base = {
      handled: false,
      keyCode,
      key,
      delta,
      tabKind: '',
      focus: false,
      shouldPreventDefault: false
    };
    if (!isDown || !event || (keyCode !== 'Tab' && key !== 'Tab')) return base;
    const target = event.target || null;
    const tabKind = getPanelTabCycleTargetKind(target);
    if (tabKind) {
      return Object.assign({}, base, {
        handled: true,
        tabKind,
        focus: true,
        shouldPreventDefault: true
      });
    }
    if (target === currentState.canvas) {
      const topWindowId = String(currentState.topWindowId || '');
      if (topWindowId === 'skills') {
        return Object.assign({}, base, {
          handled: true,
          tabKind: 'skill',
          focus: false,
          shouldPreventDefault: true
        });
      }
      if (topWindowId === 'inventory') {
        return Object.assign({}, base, {
          handled: true,
          tabKind: 'inventory',
          focus: false,
          shouldPreventDefault: true
        });
      }
      if (topWindowId === 'character') {
        return Object.assign({}, base, {
          handled: true,
          tabKind: 'character',
          focus: false,
          shouldPreventDefault: true
        });
      }
    }
    return base;
  }

  function getQuestPromptInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const keyCode = String(code || '');
    const base = {
      handled: false,
      keyCode,
      action: '',
      shouldPreventDefault: false,
      shouldConfirmQuestPrompt: false,
      shouldDeclineQuestPrompt: false
    };
    if (!isDown || !currentState.hasPrompt) return base;
    if (keyCode === 'KeyY') {
      return Object.assign({}, base, {
        handled: true,
        action: 'confirm',
        shouldPreventDefault: true,
        shouldConfirmQuestPrompt: true
      });
    }
    if (keyCode === 'KeyN') {
      return Object.assign({}, base, {
        handled: true,
        action: 'decline',
        shouldPreventDefault: true,
        shouldDeclineQuestPrompt: true
      });
    }
    return base;
  }

  function getDropQuantityPromptInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const keyCode = String(code || '');
    const base = {
      handled: false,
      keyCode,
      action: '',
      shouldPreventDefault: false,
      shouldConfirmDropQuantityPrompt: false,
      shouldCancelDropQuantityPrompt: false,
      shouldAdjustDropQuantityPrompt: false,
      adjustDelta: 0,
      shouldSetDropQuantityPromptValue: false,
      promptValue: null,
      shouldSetDropQuantityPromptMax: false
    };
    if (!isDown || !currentState.hasPrompt) return base;
    const handledBase = Object.assign({}, base, {
      handled: true,
      action: 'consume',
      shouldPreventDefault: true
    });
    if (keyCode === 'Enter' || keyCode === 'NumpadEnter') {
      return Object.assign({}, handledBase, {
        action: 'confirm',
        shouldConfirmDropQuantityPrompt: true
      });
    }
    if (keyCode === 'Escape') {
      return Object.assign({}, handledBase, {
        action: 'cancel',
        shouldCancelDropQuantityPrompt: true
      });
    }
    if (keyCode === 'ArrowLeft' || keyCode === 'ArrowDown') {
      return Object.assign({}, handledBase, {
        action: 'adjust',
        shouldAdjustDropQuantityPrompt: true,
        adjustDelta: -1
      });
    }
    if (keyCode === 'ArrowRight' || keyCode === 'ArrowUp') {
      return Object.assign({}, handledBase, {
        action: 'adjust',
        shouldAdjustDropQuantityPrompt: true,
        adjustDelta: 1
      });
    }
    if (keyCode === 'Home') {
      return Object.assign({}, handledBase, {
        action: 'setMin',
        shouldSetDropQuantityPromptValue: true,
        promptValue: 1
      });
    }
    if (keyCode === 'End') {
      return Object.assign({}, handledBase, {
        action: 'setMax',
        shouldSetDropQuantityPromptValue: true,
        shouldSetDropQuantityPromptMax: true
      });
    }
    return handledBase;
  }

  function getConfirmPromptInputMetadata(code, isDown, state) {
    const currentState = state || {};
    const keyCode = String(code || '');
    const base = {
      handled: false,
      keyCode,
      action: '',
      shouldPreventDefault: false,
      shouldConfirmPrompt: false,
      shouldCancelConfirmPrompt: false
    };
    if (!isDown || !currentState.hasPrompt) return base;
    const handledBase = Object.assign({}, base, {
      handled: true,
      action: 'consume',
      shouldPreventDefault: true
    });
    if (keyCode === 'Enter' || keyCode === 'NumpadEnter' || keyCode === 'KeyY') {
      return Object.assign({}, handledBase, {
        action: 'confirm',
        shouldConfirmPrompt: true
      });
    }
    if (keyCode === 'Escape' || keyCode === 'KeyN') {
      return Object.assign({}, handledBase, {
        action: 'cancel',
        shouldCancelConfirmPrompt: true
      });
    }
    return handledBase;
  }

  function getMonsterGuideSearchInputMetadata(event, isDown, state) {
    const currentState = state || {};
    const keyCode = event && event.code || '';
    const key = event && event.key || '';
    const base = {
      handled: false,
      keyCode,
      key,
      action: '',
      shouldPreventDefault: false,
      shouldSetQuery: false,
      query: '',
      keepScroll: false,
      shouldBlurSearch: false,
      shouldRequestCanvasDraw: false
    };
    if (!currentState.searchFocused) return base;
    if (!isDown) {
      return Object.assign({}, base, {
        handled: true,
        action: 'consume'
      });
    }
    const handledBase = Object.assign({}, base, {
      handled: true,
      action: 'consume',
      shouldPreventDefault: true
    });
    const query = String(currentState.query || '');
    if (keyCode === 'Escape') {
      if (query) {
        return Object.assign({}, handledBase, {
          action: 'clear',
          shouldSetQuery: true,
          query: ''
        });
      }
      return Object.assign({}, handledBase, {
        action: 'blur',
        shouldBlurSearch: true,
        shouldRequestCanvasDraw: true
      });
    }
    if (keyCode === 'Enter' || keyCode === 'NumpadEnter' || keyCode === 'Tab') {
      return Object.assign({}, handledBase, {
        action: 'blur',
        shouldBlurSearch: true,
        shouldRequestCanvasDraw: true
      });
    }
    if (keyCode === 'Backspace') {
      return Object.assign({}, handledBase, {
        action: 'backspace',
        shouldSetQuery: true,
        query: query.slice(0, -1)
      });
    }
    if (keyCode === 'Delete') {
      return Object.assign({}, handledBase, {
        action: 'delete',
        shouldSetQuery: true,
        query: ''
      });
    }
    if (key && key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      return Object.assign({}, handledBase, {
        action: 'append',
        shouldSetQuery: true,
        query: `${query}${key}`
      });
    }
    return handledBase;
  }

  function getGlobalEscapeInputMetadata(event, isDown) {
    const code = event && event.code || '';
    const handled = !!isDown && code === 'Escape';
    return {
      handled,
      shouldPreventDefault: handled,
      shouldHandleEscapeMenu: handled
    };
  }

  function getEscapeMenuInputAction(state) {
    const currentState = state || {};
    const base = {
      handled: true,
      action: '',
      needsBlockingState: false
    };
    if (currentState.isCommandOpen) return Object.assign({}, base, { action: 'closeCommandPanel' });
    if (currentState.hasDropQuantityPrompt) return Object.assign({}, base, { action: 'cancelDropQuantityPrompt' });
    if (currentState.hasAdminNumberPrompt) return Object.assign({}, base, { action: 'cancelAdminNumberPrompt' });
    if (currentState.hasConfirmPrompt) return Object.assign({}, base, { action: 'cancelConfirmPrompt' });
    if (currentState.hasPendingInventoryDrop) return Object.assign({}, base, { action: 'clearPendingInventoryDrop' });
    if (currentState.hasGearPickerContext) return Object.assign({}, base, { action: 'closeGearPicker' });
    if (currentState.potentialPromptOpen) return Object.assign({}, base, { action: 'closePotentialPrompt' });
    if (currentState.shardCraftPromptOpen) return Object.assign({}, base, { action: 'closeShardCraftPrompt' });
    if (currentState.upgradePromptOpen) return Object.assign({}, base, { action: 'closeUpgradePrompt' });
    if (Number(currentState.openWindowCount || 0) > 0) return Object.assign({}, base, { action: 'closeTopWindow' });
    if (!Object.prototype.hasOwnProperty.call(currentState, 'hasBlockingGameUiOpen')) {
      return Object.assign({}, base, {
        action: 'resolveBlockingState',
        needsBlockingState: true
      });
    }
    if (currentState.hasBlockingGameUiOpen) return Object.assign({}, base, { action: 'noopBlockingGameUi' });
    return Object.assign({}, base, { action: 'openCommandPanel' });
  }

  function createInputUiHelpers() {
    return Object.freeze({
      getAttackKeyInputMetadata,
      getAttackKeyInputStateAction,
      getSkillKeyInputMetadata,
      getSkillKeyInputStateAction,
      getClearHoldInputsMetadata,
      getClearHoldInputsStateAction,
      getReleaseAttackInputMetadata,
      getReleaseAttackInputStateAction,
      getDomPointerReleaseCleanupMetadata,
      getDomAttackButtonPointerAction,
      getTouchControlTarget,
      getTouchControlPointerAction,
      getTouchControlReleaseAction,
      getOutOfRootClickAction,
      getDomClickTarget,
      getDomInputTarget,
      getFixedMovementInputMetadata,
      getBoundActionInputDispatch,
      getBoundActionsInputMetadata,
      getCommandPanelTarget,
      getGameKeyTargetMetadata,
      getModalKeyboardInputMetadata,
      getAdjacentModalControlFocusMetadata,
      getPanelTabCycleTargetKind,
      getPanelTabCycleInputMetadata,
      getQuestPromptInputMetadata,
      getDropQuantityPromptInputMetadata,
      getConfirmPromptInputMetadata,
      getMonsterGuideSearchInputMetadata,
      getGlobalEscapeInputMetadata,
      getEscapeMenuInputAction
    });
  }

  function createInputDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_CLICK_TARGET_ATTRIBUTES,
      DOM_CLICK_TARGET_SELECTOR,
      DOM_INPUT_TARGET_ATTRIBUTES,
      DOM_INPUT_TARGET_SELECTOR,
      DOM_SKILL_TAB_TARGET_ATTRIBUTES,
      DOM_SKILL_TAB_TARGET_SELECTOR,
      DOM_INVENTORY_TAB_TARGET_ATTRIBUTES,
      DOM_INVENTORY_TAB_TARGET_SELECTOR,
      DOM_CHARACTER_TAB_TARGET_ATTRIBUTES,
      DOM_CHARACTER_TAB_TARGET_SELECTOR,
      DOM_COMMAND_PANEL_TARGET_ATTRIBUTES,
      DOM_COMMAND_PANEL_TARGET_SELECTOR,
      DOM_TOUCH_CONTROL_TARGET_ATTRIBUTES,
      DOM_TOUCH_CONTROL_TARGET_SELECTOR
    });
  }

  const api = {
    DOM_CLICK_TARGET_ATTRIBUTES,
    DOM_CLICK_TARGET_SELECTOR,
    DOM_INPUT_TARGET_ATTRIBUTES,
    DOM_INPUT_TARGET_SELECTOR,
    DOM_SKILL_TAB_TARGET_ATTRIBUTES,
    DOM_SKILL_TAB_TARGET_SELECTOR,
    DOM_INVENTORY_TAB_TARGET_ATTRIBUTES,
    DOM_INVENTORY_TAB_TARGET_SELECTOR,
    DOM_CHARACTER_TAB_TARGET_ATTRIBUTES,
    DOM_CHARACTER_TAB_TARGET_SELECTOR,
    DOM_COMMAND_PANEL_TARGET_ATTRIBUTES,
    DOM_COMMAND_PANEL_TARGET_SELECTOR,
    DOM_TOUCH_CONTROL_TARGET_ATTRIBUTES,
    DOM_TOUCH_CONTROL_TARGET_SELECTOR,
    getAttackKeyInputMetadata,
    getAttackKeyInputStateAction,
    getSkillKeyInputMetadata,
    getSkillKeyInputStateAction,
    getClearHoldInputsMetadata,
    getClearHoldInputsStateAction,
    getReleaseAttackInputMetadata,
    getReleaseAttackInputStateAction,
    getDomPointerReleaseCleanupMetadata,
    getDomAttackButtonPointerAction,
    getTouchControlTarget,
    getTouchControlPointerAction,
    getTouchControlReleaseAction,
    getOutOfRootClickAction,
    getDomClickTarget,
    getDomInputTarget,
    getFixedMovementInputMetadata,
    getBoundActionInputDispatch,
    getBoundActionsInputMetadata,
    getCommandPanelTarget,
    getGameKeyTargetMetadata,
    getModalKeyboardInputMetadata,
    getAdjacentModalControlFocusMetadata,
    getPanelTabCycleTargetKind,
    getPanelTabCycleInputMetadata,
    getQuestPromptInputMetadata,
    getDropQuantityPromptInputMetadata,
    getConfirmPromptInputMetadata,
    getMonsterGuideSearchInputMetadata,
    getGlobalEscapeInputMetadata,
    getEscapeMenuInputAction,
    createInputUiHelpers,
    createInputDomSelectorUiHelpers
  };

  global.ProjectStarfallUiModules = global.ProjectStarfallUiModules || {};
  global.ProjectStarfallUiModules.input = api;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
