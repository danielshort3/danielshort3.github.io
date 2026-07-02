(function initProjectStarfallUiKeybindings(global) {
  'use strict';

  const KEYBIND_STORAGE_KEY = 'projectStarfallKeybinds.v4';
  const FIXED_MOVEMENT_KEYS = Object.freeze({
    ArrowLeft: 'left',
    ArrowRight: 'right',
    ArrowUp: 'up',
    ArrowDown: 'down'
  });

  function createFixedMovementKeyCodes(keys) {
    return Object.freeze(Object.keys(keys && typeof keys === 'object' ? keys : {}));
  }

  const FIXED_MOVEMENT_KEY_CODES = createFixedMovementKeyCodes(FIXED_MOVEMENT_KEYS);
  const KEYBIND_ACTIONS = Object.freeze([
    { id: 'jump', label: 'Jump', type: 'hold', input: 'jump', defaultKeys: ['Space'] },
    { id: 'attack', label: 'Attack', type: 'action', action: 'attack', defaultKeys: ['KeyJ', 'ShiftLeft'] },
    { id: 'party', label: 'Party Skill', type: 'action', action: 'party', defaultKeys: ['KeyL'] },
    { id: 'interact', label: 'Interact', type: 'action', action: 'interact', defaultKeys: ['KeyF'] },
    { id: 'npcTalk', label: 'Talk to NPC', type: 'action', action: 'npcTalk', defaultKeys: ['KeyY'] },
    { id: 'loot', label: 'Loot', type: 'hold', input: 'loot', action: 'loot', defaultKeys: ['KeyZ'] },
    { id: 'menu', label: 'Menu', type: 'action', action: 'menu', defaultKeys: ['Escape'] },
    { id: 'keybinds', label: 'Keybind Menu', type: 'panel', panel: 'keybinds', defaultKeys: ['Backslash'] },
    { id: 'minimap', label: 'Minimap Compact', type: 'action', action: 'minimap', defaultKeys: ['KeyM'] },
    { id: 'character', label: 'Character Popup', type: 'panel', panel: 'character', defaultKeys: ['Comma'] },
    { id: 'equipment', label: 'Equipment Popup', type: 'panel', panel: 'equipment', defaultKeys: ['KeyE'] },
    { id: 'partyPanel', label: 'Party Popup', type: 'panel', panel: 'partyPanel', defaultKeys: ['KeyP'] },
    { id: 'pet', label: 'Pet Popup', type: 'panel', panel: 'pet', defaultKeys: ['KeyT'] },
    { id: 'worldmap', label: 'World Map', type: 'panel', panel: 'worldmap', defaultKeys: ['KeyW'] },
    { id: 'monsters', label: 'Monster Guide', type: 'panel', panel: 'monsters', defaultKeys: ['KeyN'] },
    { id: 'skills', label: 'Skills Popup', type: 'panel', panel: 'skills', defaultKeys: ['KeyK'] },
    { id: 'quests', label: 'Quest Popup', type: 'panel', panel: 'quests', defaultKeys: ['KeyQ'] },
    { id: 'inventory', label: 'Inventory Popup', type: 'panel', panel: 'inventory', defaultKeys: ['KeyI'] },
    { id: 'shop', label: 'Shop Popup', type: 'panel', panel: 'shop', defaultKeys: ['KeyO'] },
    { id: 'upgrade', label: 'Upgrade Popup', type: 'panel', panel: 'upgrade', defaultKeys: ['KeyU'] },
    { id: 'plinko', label: 'Plinko Popup', type: 'panel', panel: 'plinko', defaultKeys: [] },
    { id: 'daily', label: 'Daily Rewards Popup', type: 'panel', panel: 'daily', defaultKeys: [] },
    { id: 'attunement', label: 'Attunement Popup', type: 'action', action: 'attunement', defaultKeys: ['KeyR'] },
    { id: 'cashShop', label: 'Cash Shop Popup', type: 'panel', panel: 'cashShop', defaultKeys: [] },
    { id: 'beta', label: 'Beta Systems Popup', type: 'panel', panel: 'beta', defaultKeys: [] },
    { id: 'guide', label: 'Guide Popup', type: 'panel', panel: 'guide', defaultKeys: [] },
    { id: 'log', label: 'Session Log Popup', type: 'panel', panel: 'log', defaultKeys: ['KeyG'] },
    { id: 'save', label: 'Save Character', type: 'action', action: 'save', defaultKeys: ['F6'] },
    { id: 'load', label: 'Character Select', type: 'action', action: 'load', defaultKeys: ['F7'] },
    { id: 'reset', label: 'Delete Character', type: 'action', action: 'reset', defaultKeys: ['F8'] },
    { id: 'performanceDebug', label: 'Performance Debug', type: 'action', action: 'performanceDebug', defaultKeys: ['F3'] },
    { id: 'combatMetrics', label: 'Combat Metrics', type: 'action', action: 'combatMetrics', defaultKeys: ['F4'] },
    { id: 'boost', label: 'Level 100 Boost', type: 'action', action: 'boost', defaultKeys: ['KeyB'] },
    { id: 'settings', label: 'Settings Popup', type: 'panel', panel: 'settings', defaultKeys: ['F10'] },
    { id: 'admin', label: 'Admin Settings', type: 'panel', panel: 'admin', defaultKeys: [] }
  ]);

  function createActionById(actions) {
    return (Array.isArray(actions) ? actions : []).reduce((map, action) => {
      map[action.id] = action;
      return map;
    }, {});
  }

  const ACTION_BY_ID = createActionById(KEYBIND_ACTIONS);

  function createDefaultTrayActionIds(actions) {
    return Object.freeze((Array.isArray(actions) ? actions : []).map((action) => action.id));
  }

  const DEFAULT_TRAY_ACTION_IDS = createDefaultTrayActionIds(KEYBIND_ACTIONS);
  const KEYBIND_CORE_ACTION_GROUPS = Object.freeze(['Movement', 'Combat', 'Menu/Panels', 'System']);
  const SKILL_BIND_PREFIX = 'skill:';
  const ITEM_BIND_PREFIX = 'item:';
  const ATTACK_HOLD_DEFAULT_KEY = 'ShiftLeft';

  function createBlockedBindingKeys(fixedMovementKeyCodes) {
    const movementCodes = Array.isArray(fixedMovementKeyCodes) ? fixedMovementKeyCodes : [];
    return Object.freeze(['CapsLock', 'MetaLeft', 'MetaRight', 'Backspace', ...movementCodes]);
  }

  const BLOCKED_BINDING_KEYS = createBlockedBindingKeys(FIXED_MOVEMENT_KEY_CODES);

  function createBlockedBindingKeySet(keys) {
    return new Set(Array.isArray(keys) ? keys : []);
  }

  const BLOCKED_BINDING_KEY_SET = createBlockedBindingKeySet(BLOCKED_BINDING_KEYS);
  const KEYBIND_CLICK_ATTRIBUTES = Object.freeze([
    'data-starfall-rebind',
    'data-starfall-bind-action',
    'data-starfall-key-target',
    'data-starfall-disabled-key',
    'data-starfall-clear-key',
    'data-starfall-reset-keybinds'
  ]);
  const DOM_BOUND_ACTION_DRAG_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-bound-action'
  ]);
  const DOM_BOUND_ACTION_DRAG_TARGET_SELECTOR = DOM_BOUND_ACTION_DRAG_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_BIND_ACTION_DRAG_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-bind-action'
  ]);
  const DOM_BIND_ACTION_DRAG_TARGET_SELECTOR = DOM_BIND_ACTION_DRAG_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const KEYBIND_CANVAS_REGION_TYPES = Object.freeze([
    'bind-action',
    'key-target',
    'disabled-key',
    'reset-keybinds',
    'capture-bind'
  ]);

  const KEY_LABELS = Object.freeze({
    ArrowLeft: 'Left',
    ArrowRight: 'Right',
    ArrowUp: 'Up',
    ArrowDown: 'Down',
    Space: 'Space',
    ControlLeft: 'L Ctrl',
    ControlRight: 'R Ctrl',
    ShiftLeft: 'L Shift',
    ShiftRight: 'R Shift',
    AltLeft: 'L Alt',
    AltRight: 'R Alt',
    MetaLeft: 'L Meta',
    MetaRight: 'R Meta',
    Backquote: '`',
    Minus: '-',
    Equal: '=',
    Backspace: 'Backspace',
    Tab: 'Tab',
    BracketLeft: '[',
    BracketRight: ']',
    Backslash: '\\',
    CapsLock: 'Caps',
    Semicolon: ';',
    Quote: '\'',
    Enter: 'Enter',
    Comma: ',',
    Period: '.',
    Slash: '?',
    Escape: 'Esc'
  });

  const KEYBOARD_ROWS = Object.freeze([
    { offset: 0, keys: [
      { code: 'Escape', label: 'Esc', units: 1.15, gapAfter: 0.7 },
      { code: 'F1', label: 'F1' }, { code: 'F2', label: 'F2' }, { code: 'F3', label: 'F3' }, { code: 'F4', label: 'F4', gapAfter: 0.45 },
      { code: 'F5', label: 'F5' }, { code: 'F6', label: 'F6' }, { code: 'F7', label: 'F7' }, { code: 'F8', label: 'F8', gapAfter: 0.45 },
      { code: 'F9', label: 'F9' }, { code: 'F10', label: 'F10' }, { code: 'F11', label: 'F11' }, { code: 'F12', label: 'F12' }
    ] },
    { offset: 0, keys: [
      { code: 'Backquote', label: '`' }, { code: 'Digit1', label: '1' }, { code: 'Digit2', label: '2' }, { code: 'Digit3', label: '3' },
      { code: 'Digit4', label: '4' }, { code: 'Digit5', label: '5' }, { code: 'Digit6', label: '6' }, { code: 'Digit7', label: '7' },
      { code: 'Digit8', label: '8' }, { code: 'Digit9', label: '9' }, { code: 'Digit0', label: '0' }, { code: 'Minus', label: '-' },
      { code: 'Equal', label: '=' }, { code: 'Backspace', label: 'Backspace', units: 2 }
    ] },
    { offset: 0.35, keys: [
      { code: 'Tab', label: 'Tab', units: 1.5 }, { code: 'KeyQ', label: 'Q' }, { code: 'KeyW', label: 'W' }, { code: 'KeyE', label: 'E' },
      { code: 'KeyR', label: 'R' }, { code: 'KeyT', label: 'T' }, { code: 'KeyY', label: 'Y' }, { code: 'KeyU', label: 'U' },
      { code: 'KeyI', label: 'I' }, { code: 'KeyO', label: 'O' }, { code: 'KeyP', label: 'P' }, { code: 'BracketLeft', label: '[' },
      { code: 'BracketRight', label: ']' }, { code: 'Backslash', label: '\\', units: 1.5 }
    ] },
    { offset: 0.65, keys: [
      { code: 'CapsLock', label: 'Caps', units: 1.75 }, { code: 'KeyA', label: 'A' }, { code: 'KeyS', label: 'S' }, { code: 'KeyD', label: 'D' },
      { code: 'KeyF', label: 'F' }, { code: 'KeyG', label: 'G' }, { code: 'KeyH', label: 'H' }, { code: 'KeyJ', label: 'J' },
      { code: 'KeyK', label: 'K' }, { code: 'KeyL', label: 'L' }, { code: 'Semicolon', label: ';' }, { code: 'Quote', label: '\'' },
      { code: 'Enter', label: 'Enter', units: 2.25 }
    ] },
    { offset: 1, keys: [
      { code: 'ShiftLeft', label: 'Shift', units: 2.25 }, { code: 'KeyZ', label: 'Z' }, { code: 'KeyX', label: 'X' }, { code: 'KeyC', label: 'C' },
      { code: 'KeyV', label: 'V' }, { code: 'KeyB', label: 'B' }, { code: 'KeyN', label: 'N' }, { code: 'KeyM', label: 'M' },
      { code: 'Comma', label: ',' }, { code: 'Period', label: '.' }, { code: 'Slash', label: '?' }, { code: 'ShiftRight', label: 'Shift', units: 2.75 }
    ] },
    { offset: 0.35, keys: [
      { code: 'ControlLeft', label: 'Ctrl', units: 1.4 }, { code: 'MetaLeft', label: 'Meta', units: 1.35 }, { code: 'AltLeft', label: 'Alt', units: 1.35 },
      { code: 'Space', label: 'Space', units: 6.2 }, { code: 'AltRight', label: 'Alt', units: 1.35 }, { code: 'MetaRight', label: 'Meta', units: 1.35 },
      { code: 'ControlRight', label: 'Ctrl', units: 1.4 }
    ] }
  ]);

  const ACTION_ICONS = Object.freeze({
    jump: '^',
    attack: 'ATK',
    party: 'PTY',
    interact: 'USE',
    npcTalk: 'TALK',
    loot: 'LOT',
    menu: 'MNU',
    keybinds: 'KEY',
    minimap: 'MIN',
    character: 'CHR',
    equipment: 'EQP',
    partyPanel: 'PAR',
    pet: 'PET',
    worldmap: 'MAP',
    monsters: 'MON',
    skills: 'SKL',
    quests: 'QST',
    inventory: 'BAG',
    shop: '$',
    upgrade: '+',
    plinko: 'PLK',
    daily: 'DAY',
    cashShop: 'CSH',
    attunement: 'ATN',
    guide: 'GUI',
    log: 'LOG',
    settings: 'SET',
    performanceDebug: 'FPS',
    combatMetrics: 'DPS',
    save: 'SAV',
    load: 'LOD',
    logout: 'OUT',
    reset: 'RST',
    boost: '100',
    beta: 'BETA',
    admin: 'ADM',
    assetPreview: 'AST'
  });

  function getActionIcon(action, options) {
    const icons = options && options.actionIcons || ACTION_ICONS;
    if (!action) return '';
    if (action.type === 'skill') return '';
    if (action.type === 'item') return action.asset ? '' : action.icon || 'ITM';
    return icons[action.id] || String(action.label || '?').slice(0, 3).toUpperCase();
  }

  function getCanvasActionIconMetadata(action, x, y, size, options) {
    if (action && action.type === 'skill') {
      return {
        kind: 'skill',
        skillId: action.skillId,
        x,
        y,
        size,
        cooldownSkillId: action.skillId,
        cooldownMinSize: 18
      };
    }
    if (action && action.type === 'item') {
      return {
        kind: 'item',
        itemId: action.itemId,
        x,
        y,
        size,
        iconOptions: { showAura: false }
      };
    }
    return {
      kind: 'badge',
      label: getActionIcon(action, options),
      x,
      y,
      size,
      fill: '#f6f0dc'
    };
  }

  function drawCanvasActionIcon(ctx, action, x, y, size, options) {
    const settings = options || {};
    const icon = getCanvasActionIconMetadata(action, x, y, size, { actionIcons: settings.actionIcons });
    if (icon.kind === 'skill') {
      if (typeof settings.drawSkillIcon !== 'function') return false;
      settings.drawSkillIcon({
        skillId: icon.skillId,
        x: icon.x,
        y: icon.y,
        size: icon.size
      });
      const cooldown = typeof settings.getActiveSkillCooldown === 'function'
        ? settings.getActiveSkillCooldown(icon.cooldownSkillId)
        : null;
      if (
        cooldown &&
        icon.size >= icon.cooldownMinSize &&
        typeof settings.drawCanvasSkillCooldownOverlay === 'function'
      ) {
        settings.drawCanvasSkillCooldownOverlay({
          x: icon.x,
          y: icon.y,
          size: icon.size,
          cooldown
        });
      }
      return true;
    }
    if (icon.kind === 'item') {
      if (typeof settings.drawItemIcon !== 'function') return false;
      settings.drawItemIcon({
        itemId: icon.itemId,
        x: icon.x,
        y: icon.y,
        size: icon.size,
        options: icon.iconOptions
      });
      return true;
    }
    if (typeof settings.drawIconBadge !== 'function') return false;
    settings.drawIconBadge({
      label: icon.label,
      x: icon.x,
      y: icon.y,
      size: icon.size,
      fill: icon.fill,
      stroke: icon.stroke
    });
    return true;
  }

  function iconClass(value) {
    return String(value || 'slash').replace(/[^a-z0-9-]/gi, '') || 'slash';
  }

  function escapeHtmlFallback(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderSkillIconMarkup(skill, className, options) {
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const getIconClass = settings.iconClass || iconClass;
    const iconKind = getIconClass(skill && skill.iconKind);
    const asset = skill && skill.iconAsset ? String(skill.iconAsset) : '';
    const classes = ['project-starfall-skill-icon', `project-starfall-skill-icon--${iconKind}`, asset ? 'has-image' : '', className || ''].filter(Boolean).join(' ');
    return `
      <span class="${escapeHtml(classes)}" aria-hidden="true">
        ${asset ? `<img src="${escapeHtml(asset)}" alt="" loading="lazy" decoding="async">` : ''}
      </span>
    `;
  }

  function renderActionIconMarkup(action, className, options) {
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const getSkillById = settings.getSkillById || function getSkillByIdFallback() { return null; };
    const getIconClass = settings.iconClass || iconClass;
    const getIconLabel = settings.getActionIcon || function getActionIconFallback(entry) {
      return getActionIcon(entry, settings);
    };
    const renderAssetImage = settings.renderAssetImage || function renderAssetImageFallback() { return ''; };
    if (action && action.type === 'skill') return renderSkillIconMarkup(getSkillById(action.skillId), className, settings);
    const asset = action && action.type === 'item' ? String(action.asset || '') : '';
    const kind = getIconClass(action && (action.action || action.id || action.type));
    const label = getIconLabel(action);
    return `
      <span class="${escapeHtml(['project-starfall-action-icon', `project-starfall-action-icon--${kind}`, asset ? 'has-image' : '', className || ''].filter(Boolean).join(' '))}" aria-hidden="true">
        ${asset ? renderAssetImage(asset, '', 'project-starfall-item-art') : ''}
        ${!asset && label ? `<span>${escapeHtml(label)}</span>` : ''}
      </span>
    `;
  }

  function formatKeyCode(code) {
    const value = String(code || '');
    if (KEY_LABELS[value]) return KEY_LABELS[value];
    if (/^Key[A-Z]$/.test(value)) return value.slice(3);
    if (/^Digit[0-9]$/.test(value)) return value.slice(5);
    if (/^Numpad[0-9]$/.test(value)) return `Num ${value.slice(6)}`;
    if (/^F[0-9]+$/.test(value)) return value;
    return value.replace(/([a-z])([A-Z])/g, '$1 $2');
  }

  function getBindActionMetadata(action, state, options) {
    const entry = action || {};
    const panelState = state || {};
    const settings = options || {};
    const formatKey = typeof settings.formatKeyCode === 'function' ? settings.formatKeyCode : formatKeyCode;
    const keybinds = panelState.keybinds || {};
    const actionId = entry.id;
    const keys = (keybinds[actionId] || []).map(formatKey).join(' / ') || 'Unbound';
    const selected = panelState.selectedBindActionId === actionId;
    const rebinding = panelState.rebindingAction === actionId;
    const itemClass = entry.type === 'item' ? ' is-item-action' : '';
    const detail = settings.detail || `${entry.label} | ${keys}`;
    return {
      actionId,
      label: entry.label,
      keys,
      selected,
      rebinding,
      itemClass,
      className: `project-starfall-bind-action${itemClass} ${selected ? 'is-selected' : ''} ${rebinding ? 'is-rebinding' : ''}`,
      detail,
      buttonLabel: rebinding ? 'Press key' : 'Capture'
    };
  }

  function getVirtualKeyMetadata(key, ownerId, ownerLabel, ownerAction) {
    const entry = key || {};
    const code = String(entry.code || '');
    const label = entry.label || formatKeyCode(code);
    const owner = String(ownerId || '');
    const boundLabel = String(ownerLabel || '');
    const sizeClass = entry.units && entry.units > 2
      ? ' project-starfall-key--space'
      : entry.units && entry.units > 1
        ? ' project-starfall-key--wide'
        : '';
    const disabled = !isAssignableKeyCode(code);
    return {
      code,
      label,
      ownerId: owner,
      ownerLabel: boundLabel,
      ownerAction: ownerAction || null,
      sizeClass,
      disabled,
      bound: !!owner,
      draggable: !!(owner && !disabled),
      className: `project-starfall-key${sizeClass} ${owner ? 'is-bound' : ''} ${disabled ? 'is-disabled' : ''}`,
      targetKind: disabled ? 'disabled' : 'target',
      ariaLabel: `${label}${boundLabel ? ` bound to ${boundLabel}` : ''}`,
      ownerIconClass: ownerAction && ownerAction.type === 'item' ? 'project-starfall-key-action-icon is-item-action' : 'project-starfall-key-action-icon',
      clearLabel: `Clear ${label}`,
      lockedText: 'Locked',
      dropText: 'Drop here'
    };
  }

  function getKeybindLookupCacheMetadata(keybinds, revision, cached, options) {
    const bindings = keybinds || {};
    const normalizedRevision = Math.max(0, Math.floor(Number(revision || 0) || 0));
    if (cached && cached.keybindsRef === bindings && cached.revision === normalizedRevision) {
      return {
        cache: cached,
        reused: true
      };
    }
    const settings = options || {};
    const getBindableAction = typeof settings.getBindableAction === 'function'
      ? settings.getBindableAction
      : function getBindableActionFallback() { return null; };
    const actionsByCode = new Map();
    const ownerIdByCode = new Map();
    Object.keys(bindings).forEach((actionId) => {
      const codes = bindings[actionId] || [];
      const action = getBindableAction(actionId);
      for (let codeIndex = 0; codeIndex < codes.length; codeIndex += 1) {
        const code = String(codes[codeIndex] || '');
        if (!code) continue;
        if (!ownerIdByCode.has(code)) ownerIdByCode.set(code, actionId);
        if (!action) continue;
        if (!actionsByCode.has(code)) actionsByCode.set(code, []);
        actionsByCode.get(code).push(action);
      }
    });
    return {
      cache: {
        keybindsRef: bindings,
        revision: normalizedRevision,
        actionsByCode,
        ownerIdByCode
      },
      reused: false
    };
  }

  function getInvalidateKeybindLookupCacheStateAction(state) {
    const currentState = state || {};
    const revision = Math.max(0, Math.floor(Number(currentState.keybindLookupRevision || 0) || 0)) + 1;
    return {
      shouldSetKeybindLookupCache: true,
      keybindLookupCache: null,
      shouldSetKeybindLookupRevision: true,
      keybindLookupRevision: revision,
      shouldDeleteBindActionCacheKey: true,
      bindActionCacheKey: 'unassigned'
    };
  }

  function getCoreActionGroup(action) {
    const entry = action || {};
    if (entry.id === 'jump') return 'Movement';
    if (['attack', 'party', 'interact', 'npcTalk', 'loot'].includes(entry.id)) return 'Combat';
    if (entry.id === 'menu' || entry.type === 'panel') return 'Menu/Panels';
    return 'System';
  }

  function getCoreBindActions(actions) {
    const source = Array.isArray(actions) ? actions : KEYBIND_ACTIONS;
    return source.map((action) => Object.assign({}, action, {
      group: getCoreActionGroup(action)
    }));
  }

  function getDefaultActionGroups(actions, groups) {
    const source = Array.isArray(actions) ? actions : [];
    const groupOrder = Array.isArray(groups) ? groups : KEYBIND_CORE_ACTION_GROUPS;
    return groupOrder
      .map((group) => ({
        group,
        actions: source.filter((action) => action && action.group === group)
      }))
      .filter((group) => group.actions.length);
  }

  function getBindableActionMetadata(actionId, options) {
    const id = String(actionId || '');
    const settings = options || {};
    const actionById = settings.actionById || ACTION_BY_ID;
    const coreAction = actionById[id];
    if (coreAction) {
      const getGroup = typeof settings.getCoreActionGroup === 'function'
        ? settings.getCoreActionGroup
        : getCoreActionGroup;
      return Object.assign({}, coreAction, {
        group: getGroup(coreAction)
      });
    }
    if (id.startsWith(SKILL_BIND_PREFIX)) {
      const skillId = id.slice(SKILL_BIND_PREFIX.length);
      const getSkillById = typeof settings.getSkillById === 'function'
        ? settings.getSkillById
        : function getSkillByIdFallback() { return null; };
      const skill = getSkillById(skillId);
      return skill ? { id, label: skill.name, type: 'skill', skillId, group: 'Skills' } : null;
    }
    if (id.startsWith(ITEM_BIND_PREFIX)) {
      const itemId = id.slice(ITEM_BIND_PREFIX.length);
      const getConsumableItem = typeof settings.getConsumableItem === 'function'
        ? settings.getConsumableItem
        : function getConsumableItemFallback() { return null; };
      const getItemAsset = typeof settings.getItemAsset === 'function'
        ? settings.getItemAsset
        : function getItemAssetFallback() { return null; };
      const item = getConsumableItem(itemId);
      return item ? {
        id,
        label: item.name,
        type: 'item',
        itemId,
        icon: item.icon || 'ITM',
        asset: getItemAsset(item),
        group: 'Items'
      } : null;
    }
    return null;
  }

  function getDefaultBindActions(options) {
    const settings = options || {};
    const actionIds = Array.isArray(settings.defaultTrayActionIds)
      ? settings.defaultTrayActionIds
      : DEFAULT_TRAY_ACTION_IDS;
    const getBindableAction = typeof settings.getBindableAction === 'function'
      ? settings.getBindableAction
      : function getBindableActionFallback(actionId) {
        return getBindableActionMetadata(actionId, settings);
      };
    return actionIds
      .map((id) => getBindableAction(id))
      .filter(Boolean);
  }

  function getSkillBindActionsMetadata(snapshot, cached, options) {
    const snapshotData = snapshot || {};
    const state = snapshotData.state || {};
    const player = state.player || {};
    const ranks = state.skills || {};
    const revisions = snapshotData.domainRevisions || {};
    const settings = options || {};
    const hasKeys = typeof settings.hasEnumerableKeys === 'function'
      ? settings.hasEnumerableKeys
      : function hasEnumerableKeysFallback(value) { return !!(value && Object.keys(value).length); };
    const isPassive = typeof settings.isPassiveSkill === 'function'
      ? settings.isPassiveSkill
      : function isPassiveSkillFallback() { return false; };
    const isUnlocked = typeof settings.isSkillUnlocked === 'function'
      ? settings.isSkillUnlocked
      : function isSkillUnlockedFallback() { return true; };
    const revisionKey = hasKeys(revisions)
      ? `domains:${Number(revisions.skills || 0)}:${Number(revisions.player || 0)}:${Number(revisions.inventory || 0)}:${Number(revisions.equipment || 0)}`
      : `snapshot:${Number(snapshotData.cacheRevision || 0)}`;
    const rankKey = Number(snapshotData.cacheRevision || 0) || hasKeys(revisions)
      ? ''
      : Object.keys(ranks)
        .filter((id) => Number(ranks[id] || 0) > 0)
        .sort()
        .map((id) => `${id}:${Number(ranks[id] || 0)}`)
        .join(',');
    const key = [
      revisionKey,
      snapshotData.skills || null,
      player.classId || '',
      player.advancedClassId || '',
      Number(player.baseSkillPoints || 0),
      Number(player.advancedSkillPoints || 0),
      rankKey
    ].join('|');
    if (cached && cached.key === key && cached.skillsRef === snapshotData.skills && cached.ranksRef === ranks && cached.playerRef === player) {
      return {
        actions: cached.actions,
        cache: cached,
        reused: true
      };
    }
    const actions = (snapshotData.skills || [])
      .filter((skill) => Number(ranks[skill.id] || 0) > 0)
      .filter((skill) => !isPassive(skill))
      .filter((skill) => isUnlocked(snapshotData, skill))
      .map((skill) => ({
        id: `${SKILL_BIND_PREFIX}${skill.id}`,
        label: skill.name,
        type: 'skill',
        skillId: skill.id,
        group: 'Skills'
      }));
    return {
      actions,
      cache: {
        key,
        skillsRef: snapshotData.skills,
        ranksRef: ranks,
        playerRef: player,
        actions
      },
      reused: false
    };
  }

  function getConsumableBindActionsMetadata(snapshot, counts, cached, options) {
    const snapshotData = snapshot || {};
    const itemCounts = counts || {};
    const revisions = snapshotData.domainRevisions || {};
    const settings = options || {};
    const hasKeys = typeof settings.hasEnumerableKeys === 'function'
      ? settings.hasEnumerableKeys
      : function hasEnumerableKeysFallback(value) { return !!(value && Object.keys(value).length); };
    const consumableItemsById = settings.consumableItemsById || {};
    const consumableItemOrderIndex = settings.consumableItemOrderIndex || {};
    const getOwnedConsumableIds = typeof settings.getOwnedConsumableIds === 'function'
      ? settings.getOwnedConsumableIds
      : function getOwnedConsumableIdsFallback(source) {
        return Object.keys(source || {})
          .filter((id) => Number(source[id] || 0) > 0 && consumableItemsById[id])
          .sort((a, b) => {
            const orderA = consumableItemOrderIndex[a];
            const orderB = consumableItemOrderIndex[b];
            return (orderA == null ? Number.MAX_SAFE_INTEGER : orderA) - (orderB == null ? Number.MAX_SAFE_INTEGER : orderB);
          });
      };
    const getConsumableItem = typeof settings.getConsumableItem === 'function'
      ? settings.getConsumableItem
      : function getConsumableItemFallback(itemId) { return consumableItemsById[String(itemId || '')] || null; };
    const getItemAsset = typeof settings.getItemAsset === 'function'
      ? settings.getItemAsset
      : function getItemAssetFallback() { return null; };
    const revisionKey = hasKeys(revisions)
      ? `inventory:${Number(revisions.inventory || 0)}`
      : `snapshot:${Number(snapshotData.cacheRevision || 0)}`;
    const countKey = Number(snapshotData.cacheRevision || 0) || hasKeys(revisions)
      ? ''
      : Object.keys(itemCounts)
        .filter((id) => Number(itemCounts[id] || 0) > 0 && consumableItemsById[id])
        .sort((a, b) => {
          const orderA = consumableItemOrderIndex[a];
          const orderB = consumableItemOrderIndex[b];
          return (orderA == null ? Number.MAX_SAFE_INTEGER : orderA) - (orderB == null ? Number.MAX_SAFE_INTEGER : orderB);
        })
        .map((id) => `${id}:${Number(itemCounts[id] || 0)}`)
        .join(',');
    const key = [
      revisionKey,
      itemCounts === null ? 'null' : 'counts',
      countKey
    ].join('|');
    if (cached && cached.key === key && cached.countsRef === itemCounts) {
      return {
        actions: cached.actions,
        cache: cached,
        reused: true
      };
    }
    const actions = getOwnedConsumableIds(itemCounts)
      .map((id) => getConsumableItem(id))
      .filter(Boolean)
      .map((item) => ({
        id: `${ITEM_BIND_PREFIX}${item.id}`,
        label: item.name,
        type: 'item',
        itemId: item.id,
        icon: item.icon || 'ITM',
        asset: getItemAsset(item),
        group: 'Items'
      }));
    return {
      actions,
      cache: {
        key,
        countsRef: itemCounts,
        actions
      },
      reused: false
    };
  }

  function getBindableActionsMetadata(skills, consumables, cached, options) {
    const skillActions = Array.isArray(skills) ? skills : [];
    const consumableActions = Array.isArray(consumables) ? consumables : [];
    const settings = options || {};
    const key = `${skillActions.length}:${consumableActions.length}`;
    if (cached && cached.key === key && cached.skills === skillActions && cached.consumables === consumableActions) {
      return {
        actions: cached.actions,
        cache: cached,
        reused: true
      };
    }
    const keybindActions = Array.isArray(settings.keybindActions) ? settings.keybindActions : KEYBIND_ACTIONS;
    const getCoreActions = typeof settings.getCoreBindActions === 'function'
      ? settings.getCoreBindActions
      : getCoreBindActions;
    const coreActions = getCoreActions(keybindActions);
    const actions = coreActions.concat(skillActions, consumableActions);
    return {
      actions,
      cache: {
        key,
        skills: skillActions,
        consumables: consumableActions,
        actions
      },
      reused: false
    };
  }

  function getUnassignedBindActionsMetadata(keybinds, bindableActions, cached) {
    const bindings = keybinds || {};
    const actionsSource = Array.isArray(bindableActions) ? bindableActions : [];
    const assignedKey = Object.keys(bindings)
      .filter((id) => !id.startsWith(ITEM_BIND_PREFIX) && (bindings[id] || []).length)
      .sort()
      .join('|');
    const key = `${actionsSource.length}:${assignedKey}`;
    if (cached && cached.key === key && cached.keybindsRef === bindings && cached.bindableActions === actionsSource) {
      return {
        actions: cached.actions,
        cache: cached,
        reused: true
      };
    }
    const actions = actionsSource
      .filter((action) => action && action.type !== 'item')
      .filter((action) => action && !(bindings[action.id] || []).length);
    return {
      actions,
      cache: {
        key,
        keybindsRef: bindings,
        bindableActions: actionsSource,
        actions
      },
      reused: false
    };
  }

  function getSelectBindActionMetadata(actionId, sourceCode, action) {
    const selected = !!action;
    return {
      selected,
      selectedBindActionId: selected ? actionId : '',
      selectedBindSourceCode: selected ? String(sourceCode || '') : '',
      rebindingAction: ''
    };
  }

  function getSelectBindActionStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldSetSelectedBindActionId: true,
      selectedBindActionId: String(currentMetadata.selectedBindActionId || ''),
      shouldSetSelectedBindSourceCode: true,
      selectedBindSourceCode: String(currentMetadata.selectedBindSourceCode || ''),
      shouldSetRebindingAction: true,
      rebindingAction: String(currentMetadata.rebindingAction || '')
    };
  }

  function getClearSelectedBindActionStateAction() {
    return {
      shouldSetSelectedBindActionId: true,
      selectedBindActionId: '',
      shouldSetSelectedBindSourceCode: true,
      selectedBindSourceCode: '',
      shouldSetRebindingAction: true,
      rebindingAction: ''
    };
  }

  function isKeybindClickTarget(target, attributes) {
    if (!target || typeof target.hasAttribute !== 'function') return false;
    const source = Array.isArray(attributes) ? attributes : KEYBIND_CLICK_ATTRIBUTES;
    return source.some((attribute) => target.hasAttribute(attribute));
  }

  function getSelectedBindOutsideClickAction(state) {
    const source = state || {};
    const selectedBindActionId = String(source.selectedBindActionId || '');
    const isKeybindTarget = !!source.isKeybindClickTarget;
    const shouldClearSelectedBind = !!selectedBindActionId && !isKeybindTarget;
    return {
      handled: shouldClearSelectedBind,
      type: shouldClearSelectedBind ? 'clearSelectedBindFromOutside' : '',
      selectedBindActionId,
      isKeybindClickTarget: isKeybindTarget,
      shouldClearSelectedBind
    };
  }

  function getKeybindResetDomAction(target) {
    const source = target || null;
    if (source && typeof source.hasAttribute === 'function' && source.hasAttribute('data-starfall-reset-keybinds')) {
      return { handled: true, type: 'restoreDefaults' };
    }
    return { handled: false, type: '' };
  }

  function getKeybindEditDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const clearCode = getAttribute('data-starfall-clear-key');
    if (clearCode) return { handled: true, type: 'clearKey', keyCode: clearCode };
    const disabledKey = getAttribute('data-starfall-disabled-key');
    if (disabledKey) return { handled: true, type: 'disabledKey', keyCode: disabledKey };
    const rebindId = getAttribute('data-starfall-rebind');
    if (rebindId) return { handled: true, type: 'startRebind', actionId: rebindId };
    return { handled: false, type: '' };
  }

  function getKeybindAssignmentDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const bindActionId = getAttribute('data-starfall-bind-action');
    if (bindActionId) return { handled: true, type: 'bindAction', actionId: bindActionId };
    const keyTarget = getAttribute('data-starfall-key-target');
    if (keyTarget) return { handled: true, type: 'keyTarget', keyCode: keyTarget };
    return { handled: false, type: '' };
  }

  function isKeybindCanvasRegionType(type, regionTypes) {
    const source = Array.isArray(regionTypes) ? regionTypes : KEYBIND_CANVAS_REGION_TYPES;
    return source.includes(String(type || ''));
  }

  function getClearSelectedBindFromOutsideMetadata(state) {
    const currentState = state || {};
    const selectedBindActionId = String(currentState.selectedBindActionId || '');
    const selectedBindSourceCode = String(currentState.selectedBindSourceCode || '');
    if (!selectedBindActionId) {
      return {
        handled: false,
        mode: 'none',
        sourceCode: '',
        shouldClearSelected: false,
        shouldRefresh: false
      };
    }
    if (selectedBindSourceCode) {
      return {
        handled: true,
        mode: 'clear-key',
        sourceCode: selectedBindSourceCode,
        shouldClearSelected: false,
        shouldRefresh: false
      };
    }
    return {
      handled: true,
      mode: 'clear-selection',
      sourceCode: '',
      shouldClearSelected: true,
      shouldRefresh: true
    };
  }

  function getClearSelectedBindFromOutsideStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearSelectedBindAction: !!currentMetadata.shouldClearSelected
    };
  }

  function getBindActionClickMetadata(actionId, state) {
    const currentState = state || {};
    const id = String(actionId || '');
    const clickCount = Math.max(1, Number(currentState.clickCount || 1) || 1);
    const inventoryDragId = String(currentState.inventoryDragId || '');
    const inventoryTab = String(currentState.inventoryTab || '');
    const isInventoryDrag = !!inventoryDragId;
    if (isInventoryDrag && id.startsWith(ITEM_BIND_PREFIX)) {
      return {
        mode: clickCount >= 2 ? 'activate-item' : 'ignore-inventory',
        actionId: id,
        itemId: id.slice(ITEM_BIND_PREFIX.length),
        inventoryTab,
        shouldPreventDefault: clickCount >= 2,
        shouldRefresh: false
      };
    }
    if (isInventoryDrag) {
      return {
        mode: 'ignore-inventory',
        actionId: id,
        itemId: '',
        inventoryTab,
        shouldPreventDefault: false,
        shouldRefresh: false
      };
    }
    return {
      mode: id ? 'select' : 'none',
      actionId: id,
      itemId: '',
      inventoryTab,
      shouldPreventDefault: false,
      shouldRefresh: !!id
    };
  }

  function getDomBindDragStartMetadata(actionId, sourceCode, action) {
    const id = String(actionId || '');
    const source = String(sourceCode || '');
    if (!id || !action) {
      return {
        mode: 'none',
        actionId: id,
        sourceCode: source,
        completedBindDrop: false,
        dataTransferEffectAllowed: '',
        dataTransferText: '',
        previewKind: 'action',
        shouldSelect: false,
        shouldMarkDragging: false
      };
    }
    return {
      mode: 'start',
      actionId: id,
      sourceCode: source,
      completedBindDrop: false,
      dataTransferEffectAllowed: source ? 'move' : 'copy',
      dataTransferText: id,
      previewKind: 'action',
      shouldSelect: true,
      shouldMarkDragging: true
    };
  }

  function getDomBindDragStartTargetAction(target, options) {
    const settings = options || {};
    const getBindableAction = typeof settings.getBindableAction === 'function'
      ? settings.getBindableAction
      : function getBindableActionFallback() { return null; };
    const containsTarget = typeof settings.containsTarget === 'function'
      ? settings.containsTarget
      : function containsTargetFallback() { return true; };
    const closest = target && typeof target.closest === 'function'
      ? (selector) => target.closest(selector)
      : () => null;
    const getAttribute = (element, name) => element && typeof element.getAttribute === 'function'
      ? element.getAttribute(name)
      : '';
    const hasAttribute = (element, name) => element && typeof element.hasAttribute === 'function'
      ? element.hasAttribute(name)
      : !!getAttribute(element, name);
    const emptyMetadata = getDomBindDragStartMetadata('', '', null);
    const boundTarget = closest(DOM_BOUND_ACTION_DRAG_TARGET_SELECTOR);
    if (boundTarget && hasAttribute(boundTarget, 'data-starfall-bound-action') && containsTarget(boundTarget)) {
      const actionId = getAttribute(boundTarget, 'data-starfall-bound-action');
      const sourceCode = getAttribute(boundTarget, 'data-starfall-bound-key') || '';
      const metadata = getDomBindDragStartMetadata(actionId, sourceCode, sourceCode ? getBindableAction(actionId) : null);
      return {
        handled: metadata.mode === 'start',
        type: metadata.mode === 'start' ? 'startBindDrag' : '',
        target: boundTarget,
        metadata,
        shouldReturnEarly: metadata.mode !== 'start'
      };
    }
    const actionTarget = closest(DOM_BIND_ACTION_DRAG_TARGET_SELECTOR);
    if (!actionTarget || !containsTarget(actionTarget)) {
      return {
        handled: false,
        type: '',
        target: actionTarget || null,
        metadata: emptyMetadata,
        shouldReturnEarly: false
      };
    }
    const actionId = getAttribute(actionTarget, 'data-starfall-bind-action');
    const metadata = getDomBindDragStartMetadata(actionId, '', getBindableAction(actionId));
    return {
      handled: metadata.mode === 'start',
      type: metadata.mode === 'start' ? 'startBindDrag' : '',
      target: actionTarget,
      metadata,
      shouldReturnEarly: metadata.mode !== 'start'
    };
  }

  function getDomBindDragStartTransferWriteAction(metadata) {
    const currentMetadata = metadata || {};
    const shouldWriteTransfer = currentMetadata.mode === 'start';
    return {
      shouldWriteTransfer,
      effectAllowed: currentMetadata.dataTransferEffectAllowed || '',
      entries: [
        {
          type: 'text/plain',
          value: String(currentMetadata.dataTransferText || ''),
          shouldWrite: shouldWriteTransfer
        }
      ]
    };
  }

  function getDomBindDragStartStateAction(metadata, target) {
    const currentMetadata = metadata || {};
    const isStart = currentMetadata.mode === 'start';
    const actionId = String(currentMetadata.actionId || '');
    const sourceCode = String(currentMetadata.sourceCode || '');
    return {
      shouldApply: isStart,
      shouldSetDraggingBindActionId: isStart,
      draggingBindActionId: actionId,
      shouldSetDraggingBindSourceCode: isStart,
      draggingBindSourceCode: sourceCode,
      shouldSetCompletedBindDrop: isStart,
      completedBindDrop: !!currentMetadata.completedBindDrop,
      shouldSelectAction: !!(isStart && currentMetadata.shouldSelect && actionId),
      selectActionId: actionId,
      selectActionCode: sourceCode,
      shouldSetNativeDragPreview: !!(isStart && target),
      previewTarget: target || null,
      preview: {
        actionId,
        kind: currentMetadata.previewKind || 'action'
      },
      shouldMarkDragging: !!(isStart && currentMetadata.shouldMarkDragging && target),
      draggingTarget: target || null,
      draggingClass: 'is-dragging'
    };
  }

  function getDomBindDragEndMetadata(state) {
    const currentState = state || {};
    const sourceCode = String(currentState.draggingBindSourceCode || '');
    const completedBindDrop = !!currentState.completedBindDrop;
    return {
      sourceCode,
      shouldClearSource: !!sourceCode && !completedBindDrop,
      draggingBindActionId: '',
      draggingBindSourceCode: '',
      completedBindDrop: false
    };
  }

  function getDomBindDragEndStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearSource: !!currentMetadata.shouldClearSource,
      sourceCode: String(currentMetadata.sourceCode || ''),
      shouldSetDraggingBindActionId: true,
      draggingBindActionId: String(currentMetadata.draggingBindActionId || ''),
      shouldSetDraggingBindSourceCode: true,
      draggingBindSourceCode: String(currentMetadata.draggingBindSourceCode || ''),
      shouldSetCompletedBindDrop: true,
      completedBindDrop: !!currentMetadata.completedBindDrop
    };
  }

  function getDomBindOutsideDropMetadata(state) {
    const currentState = state || {};
    const sourceCode = String(currentState.draggingBindSourceCode || '');
    return {
      sourceCode,
      shouldClearSource: !!sourceCode,
      completedBindDrop: !!sourceCode
    };
  }

  function getDomBindOutsideDropStateAction(metadata) {
    const currentMetadata = metadata || {};
    const shouldClearSource = !!currentMetadata.shouldClearSource;
    return {
      shouldClearSource,
      sourceCode: String(currentMetadata.sourceCode || ''),
      shouldSetCompletedBindDrop: shouldClearSource,
      completedBindDrop: !!currentMetadata.completedBindDrop
    };
  }

  function getDomBindDropTransferReadAction(inventoryPayload) {
    const payload = inventoryPayload || null;
    return {
      shouldReadText: true,
      textDataType: 'text/plain',
      isUsablePayload: !!payload && payload.tab === 'usable'
    };
  }

  function getDomBindDropActionMetadata(inventoryPayload, state) {
    const payload = inventoryPayload || null;
    const currentState = state || {};
    const isUsablePayload = !!payload && payload.tab === 'usable';
    const payloadItemId = isUsablePayload ? String(payload.id) : '';
    const dataTransferText = String(currentState.dataTransferText || '');
    const draggingBindActionId = String(currentState.draggingBindActionId || '');
    const selectedBindActionId = String(currentState.selectedBindActionId || '');
    const actionId = isUsablePayload
      ? `${ITEM_BIND_PREFIX}${payloadItemId}`
      : dataTransferText || draggingBindActionId || selectedBindActionId;
    return {
      actionId,
      itemId: actionId.startsWith(ITEM_BIND_PREFIX) ? actionId.slice(ITEM_BIND_PREFIX.length) : '',
      isUsablePayload,
      completedBindDrop: true
    };
  }

  function getDomBindDropDispatchAction(dropTargetAction, bindDropMetadata) {
    const target = dropTargetAction || {};
    const metadata = bindDropMetadata || {};
    const actionId = String(metadata.actionId || '');
    const itemId = metadata.itemId != null
      ? String(metadata.itemId || '')
      : actionId.startsWith(ITEM_BIND_PREFIX) ? actionId.slice(ITEM_BIND_PREFIX.length) : '';
    const petSlot = String(target.petSlot || '');
    if (petSlot) {
      return {
        handled: true,
        type: 'assignPetPotion',
        actionId,
        itemId,
        petSlot,
        keyTarget: '',
        completedBindDrop: metadata.completedBindDrop,
        failureMessage: 'That potion cannot be used in this pet slot.',
        shouldHandleDragEnd: true
      };
    }
    return {
      handled: true,
      type: 'assignActionToKey',
      actionId,
      itemId,
      petSlot: '',
      keyTarget: String(target.keyTarget || ''),
      completedBindDrop: metadata.completedBindDrop,
      failureMessage: '',
      shouldHandleDragEnd: true
    };
  }

  function getDomBindDropDispatchStateAction(dispatchAction) {
    const action = dispatchAction || {};
    const shouldSetCompletedBindDrop = action.type === 'assignActionToKey';
    return {
      shouldSetCompletedBindDrop,
      completedBindDrop: shouldSetCompletedBindDrop ? !!action.completedBindDrop : false
    };
  }

  function getCanvasBindDragDropMetadata(bindDrag, target) {
    const drag = bindDrag || {};
    const hasTarget = !!target;
    const actionId = String(drag.actionId || '');
    const sourceCode = String(drag.sourceCode || '');
    const targetCode = hasTarget ? String(target.code || '') : '';
    if (hasTarget) {
      return {
        mode: 'assign',
        actionId,
        sourceCode,
        targetCode,
        shouldClearCanvasSkillClick: true
      };
    }
    if (sourceCode) {
      return {
        mode: 'clear-source',
        actionId,
        sourceCode,
        targetCode: '',
        shouldClearCanvasSkillClick: true
      };
    }
    return {
      mode: 'clear-selection',
      actionId,
      sourceCode: '',
      targetCode: '',
      shouldClearCanvasSkillClick: true
    };
  }

  function getCanvasBindDragDropStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearCanvasSkillClick: !!currentMetadata.shouldClearCanvasSkillClick,
      shouldClearSelectedBindAction: currentMetadata.mode === 'clear-selection'
    };
  }

  function getCanvasBindDragClickMetadata(skillActivation, isDoubleClick, now) {
    const activation = skillActivation || null;
    if (!activation) {
      return {
        mode: 'clear-skill-click',
        skillActivation: null,
        skillId: undefined,
        now: Number(now || 0),
        shouldClearCanvasSkillClick: true,
        shouldClearSelectedBind: false,
        shouldActivateSkill: false,
        shouldRememberCanvasSkillClick: false,
        shouldClearDrag: false,
        shouldDraw: false,
        shouldPreventDefault: false,
        shouldReturnEarly: false
      };
    }
    if (isDoubleClick) {
      return {
        mode: 'activate-skill',
        skillActivation: activation,
        skillId: activation.id,
        now: Number(now || 0),
        shouldClearCanvasSkillClick: true,
        shouldClearSelectedBind: true,
        shouldActivateSkill: true,
        shouldRememberCanvasSkillClick: false,
        shouldClearDrag: true,
        shouldDraw: true,
        shouldPreventDefault: true,
        shouldReturnEarly: true
      };
    }
    return {
      mode: 'remember-skill-click',
      skillActivation: activation,
      skillId: activation.id,
      now: Number(now || 0),
      shouldClearCanvasSkillClick: false,
      shouldClearSelectedBind: false,
      shouldActivateSkill: false,
      shouldRememberCanvasSkillClick: true,
      shouldClearDrag: false,
      shouldDraw: false,
      shouldPreventDefault: false,
      shouldReturnEarly: false
    };
  }

  function getCanvasBindDragClickStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearCanvasSkillClick: !!currentMetadata.shouldClearCanvasSkillClick,
      shouldClearSelectedBindAction: !!currentMetadata.shouldClearSelectedBind,
      shouldClearCanvasBindDrag: !!currentMetadata.shouldClearDrag
    };
  }

  function getCanvasBindDropPointerAction(region, dropRegion) {
    const source = region || {};
    if (!dropRegion || source.type !== 'bind-action') {
      return {
        handled: false,
        type: '',
        dropRegion: null,
        shouldPreventDefault: false
      };
    }
    return {
      handled: true,
      type: 'captureDropRegion',
      dropRegion,
      shouldPreventDefault: true
    };
  }

  function getCanvasBindDragStartMetadata(region, point) {
    const currentRegion = region || {};
    const currentPoint = point || {};
    const startX = Number(currentPoint.x || 0);
    const startY = Number(currentPoint.y || 0);
    if (currentRegion.type === 'bind-action') {
      const actionId = String(currentRegion.actionId || '');
      return {
        mode: 'start',
        actionId,
        sourceCode: '',
        canvasBindDrag: { actionId, sourceCode: '', moved: false, startX, startY },
        shouldPreventDefault: true
      };
    }
    if (currentRegion.type === 'key-target' && currentRegion.ownerId) {
      const actionId = String(currentRegion.ownerId || '');
      const sourceCode = String(currentRegion.code || '');
      return {
        mode: 'start',
        actionId,
        sourceCode,
        canvasBindDrag: { actionId, sourceCode, moved: false, startX, startY },
        shouldPreventDefault: true
      };
    }
    return {
      mode: 'none',
      actionId: '',
      sourceCode: '',
      canvasBindDrag: null,
      shouldPreventDefault: false
    };
  }

  function getCanvasBindDragMoveMetadata(bindDrag, point, options) {
    const drag = bindDrag || {};
    const currentPoint = point || {};
    const settings = options || {};
    const configuredThreshold = Object.prototype.hasOwnProperty.call(settings, 'threshold')
      ? Number(settings.threshold)
      : 4;
    const threshold = Number.isFinite(configuredThreshold) ? configuredThreshold : 4;
    const distance = Math.hypot(
      Number(currentPoint.x) - Number(drag.startX),
      Number(currentPoint.y) - Number(drag.startY)
    );
    const shouldMarkMoved = Number.isFinite(distance) && distance > threshold;
    return {
      moved: !!drag.moved || shouldMarkMoved,
      shouldMarkMoved,
      distance,
      threshold,
      shouldPreventDefault: true,
      shouldDraw: true
    };
  }

  function getCanvasBindDragPreviewMetadata(bindDrag, pointer, bounds) {
    if (!bindDrag) return null;
    const currentPointer = pointer || {};
    const settings = bounds || {};
    const uiBottom = Number(settings.uiBottom || 0);
    const x = Number(currentPointer.x || 0) + 10;
    const maxY = Math.max(10, uiBottom - 54);
    const rawY = Number(currentPointer.y || 0) + 10;
    const y = Math.max(10, Math.min(maxY, rawY));
    return {
      actionId: String(bindDrag.actionId || ''),
      frame: {
        x,
        y,
        w: 44,
        h: 44,
        radius: 9,
        fill: 'rgba(238,246,255,0.94)',
        stroke: 'rgba(47,125,214,0.72)'
      },
      icon: {
        x: x + 6,
        y: y + 6,
        size: 32
      },
      alpha: 0.68
    };
  }

  function getCanvasBindActionGridMetadata(actions, state, layout) {
    const source = Array.isArray(actions) ? actions : [];
    const currentState = state || {};
    const settings = layout || {};
    const x = Number(settings.x || 0);
    const y = Number(settings.y || 0);
    const w = Number(settings.w || 0);
    const gap = Number(settings.gap == null ? 8 : settings.gap);
    const columnBasis = Number(settings.columnBasis == null ? 136 : settings.columnBasis);
    const minColumns = Math.max(1, Math.floor(Number(settings.minColumns == null ? 2 : settings.minColumns) || 2));
    const maxColumns = Math.max(minColumns, Math.floor(Number(settings.maxColumns == null ? 5 : settings.maxColumns) || 5));
    const safeColumnBasis = Math.max(1, columnBasis || 136);
    const actionColumns = Math.max(minColumns, Math.min(maxColumns, Math.floor(w / safeColumnBasis)));
    const cellW = Math.floor((w - gap * (actionColumns - 1)) / actionColumns);
    const rowH = Number(settings.rowHeight == null ? 38 : settings.rowHeight);
    const cardH = Number(settings.cardHeight == null ? 32 : settings.cardHeight);
    const paddingTop = Number(settings.paddingTop == null ? 8 : settings.paddingTop);
    const headerH = Number(settings.headerHeight == null ? 14 : settings.headerHeight);
    const rows = Math.max(1, Math.ceil(source.length / actionColumns));
    const groupH = headerH + rows * rowH;
    const selectedBindActionId = currentState.selectedBindActionId;
    return {
      x,
      y,
      w,
      h: groupH,
      gap,
      columnCount: actionColumns,
      cellW,
      rowCount: rows,
      rowH,
      cardH,
      isEmpty: !source.length,
      emptyText: 'All available buttons are assigned.',
      actions: source.map((action, index) => {
        const bx = x + (index % actionColumns) * (cellW + gap);
        const by = y + paddingTop + Math.floor(index / actionColumns) * rowH;
        const isItemAction = !!(action && action.type === 'item');
        const iconSize = isItemAction ? 28 : 20;
        const actionId = action && action.id;
        return {
          action,
          actionId,
          x: bx,
          y: by,
          w: cellW,
          h: cardH,
          selected: selectedBindActionId === actionId,
          isItemAction,
          iconSize,
          iconX: bx + 5,
          iconY: by + Math.floor((cardH - iconSize) / 2),
          labelX: bx + iconSize + 12,
          labelY: by + 8,
          labelMaxWidth: cellW - iconSize - 18,
          region: { type: 'bind-action', actionId, x: bx, y: by, w: cellW, h: cardH }
        };
      })
    };
  }

  function getCanvasKeyboardLayoutMetadata(rows, layout, options) {
    const source = Array.isArray(rows) ? rows : KEYBOARD_ROWS;
    const settings = layout || {};
    const hooks = options || {};
    const isAssignable = typeof hooks.isAssignableKeyCode === 'function'
      ? hooks.isAssignableKeyCode
      : isAssignableKeyCode;
    const getOwnerId = typeof hooks.getKeyOwner === 'function'
      ? hooks.getKeyOwner
      : function getKeyOwnerFallback() { return ''; };
    const getAction = typeof hooks.getBindableAction === 'function'
      ? hooks.getBindableAction
      : function getBindableActionFallback() { return null; };
    const x = Number(settings.x || 0);
    const y = Number(settings.y || 0);
    const w = Number(settings.w || 0);
    const unit = Math.max(18, Math.min(30, w / 17));
    const gap = 3;
    const keyH = 24;
    let cy = y;
    const keys = [];
    source.forEach((row) => {
      const currentRow = row || {};
      let cx = x + (currentRow.offset || 0) * unit;
      (currentRow.keys || []).forEach((key) => {
        const currentKey = key || {};
        const code = currentKey.code;
        const label = currentKey.label;
        const kw = Math.max(unit, unit * (currentKey.units || 1));
        const disabled = !isAssignable(code);
        const ownerId = getOwnerId(code);
        const action = ownerId ? getAction(ownerId) : null;
        const iconSize = action && action.type === 'item' ? 20 : 16;
        keys.push({
          key: currentKey,
          code,
          label,
          x: cx,
          y: cy,
          w: kw,
          h: keyH,
          disabled,
          ownerId,
          action,
          fill: disabled ? '#e6e3dc' : ownerId ? '#eef6ff' : '#fbfaf6',
          stroke: ownerId ? 'rgba(47,125,214,0.6)' : 'rgba(16,32,51,0.16)',
          labelColor: disabled ? '#75818c' : '#102033',
          iconSize,
          iconX: cx + kw / 2 - iconSize / 2,
          iconY: cy + 10,
          region: { type: disabled ? 'disabled-key' : 'key-target', code, ownerId, x: cx, y: cy, w: kw, h: keyH }
        });
        cx += kw + gap + (currentKey.gapAfter || 0) * unit;
      });
      cy += keyH + 5;
    });
    return {
      x,
      y,
      w,
      unit,
      gap,
      keyH,
      h: cy - y,
      keys
    };
  }

  function getAssignActionToKeyMetadata(action, code, options) {
    const keyCode = String(code || '').trim();
    const settings = options || {};
    const formatKey = typeof settings.formatKeyCode === 'function' ? settings.formatKeyCode : formatKeyCode;
    if (!action || !keyCode) {
      return {
        ok: false,
        keyCode,
        toastMessage: 'Choose an action and a key.'
      };
    }
    if (!isAssignableKeyCode(keyCode)) {
      return {
        ok: false,
        keyCode,
        toastMessage: `${formatKey(keyCode)} cannot be assigned.`
      };
    }
    return {
      ok: true,
      actionId: action.id,
      actionLabel: action.label,
      keyCode,
      selectedBindActionId: action.id,
      selectedBindSourceCode: '',
      toastMessage: `${action.label} bound to ${formatKey(keyCode)}.`
    };
  }

  function getAssignActionToKeyStateAction(metadata, action) {
    const currentMetadata = metadata || {};
    const currentAction = action || {};
    return {
      shouldSetSelectedBindActionId: true,
      selectedBindActionId: currentMetadata.selectedBindActionId == null
        ? String(currentAction.id || '')
        : String(currentMetadata.selectedBindActionId || ''),
      shouldSetSelectedBindSourceCode: true,
      selectedBindSourceCode: currentMetadata.selectedBindSourceCode == null
        ? ''
        : String(currentMetadata.selectedBindSourceCode || '')
    };
  }

  function getClearKeyBindingMetadata(code, selectedBindSourceCode, options) {
    const keyCode = String(code || '').trim();
    const settings = options || {};
    const formatKey = typeof settings.formatKeyCode === 'function' ? settings.formatKeyCode : formatKeyCode;
    if (!keyCode) {
      return {
        ok: false,
        keyCode,
        shouldClearSelected: false,
        toastMessage: ''
      };
    }
    return {
      ok: true,
      keyCode,
      shouldClearSelected: String(selectedBindSourceCode || '') === keyCode,
      toastMessage: `${formatKey(keyCode)} cleared.`
    };
  }

  function getClearKeyBindingStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldClearSelectedBindAction: !!currentMetadata.shouldClearSelected
    };
  }

  function getBindingLabel(actionId, action) {
    return action ? action.label : String(actionId || 'Unknown');
  }

  function getPrimaryKeyLabel(keybinds, actionId, options) {
    const bindings = keybinds || {};
    const settings = options || {};
    const formatKey = typeof settings.formatKeyCode === 'function' ? settings.formatKeyCode : formatKeyCode;
    const keys = bindings[actionId] || [];
    return keys.length ? formatKey(keys[0]) : 'Unbound';
  }

  function getKeyOwner(cache, code) {
    const lookup = cache && cache.ownerIdByCode;
    return lookup && lookup.get ? lookup.get(String(code || '')) || '' : '';
  }

  function getActionsForCode(cache, code) {
    if (!code) return [];
    const lookup = cache && cache.actionsByCode;
    const actions = lookup && lookup.get ? lookup.get(String(code || '')) || [] : [];
    return actions.slice();
  }

  function findBindingOwner(cache, code, exceptActionId) {
    const actions = getActionsForCode(cache, code);
    return actions.find((action) => action && action.id !== exceptActionId) || null;
  }

  function getStartRebindMetadata(actionId, action, options) {
    const settings = options || {};
    const id = String(actionId || '');
    const hasAction = !!action;
    const rebindingAction = settings.requireAction && !hasAction ? '' : id;
    const label = hasAction ? action.label : String(id || 'Unknown');
    return {
      actionId: id,
      rebindingAction,
      shouldRefresh: !!settings.refresh,
      toastMessage: settings.toast ? `Press a key for ${label}.` : ''
    };
  }

  function getStartRebindStateAction(metadata) {
    const currentMetadata = metadata || {};
    return {
      shouldSetRebindingAction: true,
      rebindingAction: String(currentMetadata.rebindingAction || '')
    };
  }

  function getKeyTargetInteractionMetadata(code, state, options) {
    const currentState = state || {};
    const settings = options || {};
    const keyCode = String(code || '');
    const selectedBindActionId = String(currentState.selectedBindActionId || '');
    const ownerId = String(currentState.ownerId || '');
    if (selectedBindActionId) {
      return {
        mode: 'assign',
        keyCode,
        actionId: selectedBindActionId,
        sourceCode: '',
        shouldRefresh: false
      };
    }
    if (ownerId) {
      return {
        mode: 'select',
        keyCode,
        actionId: ownerId,
        sourceCode: keyCode,
        shouldRefresh: !!settings.refreshOnSelect
      };
    }
    return {
      mode: 'none',
      keyCode,
      actionId: '',
      sourceCode: '',
      shouldRefresh: false
    };
  }

  function getDisabledKeyFeedbackMetadata(code, options) {
    const keyCode = String(code || '');
    const settings = options || {};
    const formatKey = typeof settings.formatKeyCode === 'function' ? settings.formatKeyCode : formatKeyCode;
    return {
      keyCode,
      toastMessage: `${formatKey(keyCode)} cannot be assigned.`
    };
  }

  function getKeyboardRebindCaptureAction(event, isDown, rebindingAction) {
    const activeActionId = String(rebindingAction || '');
    const shouldCaptureRebind = !!isDown && !!activeActionId;
    return {
      handled: shouldCaptureRebind,
      type: shouldCaptureRebind ? 'captureRebind' : '',
      actionId: activeActionId,
      shouldCaptureRebind
    };
  }

  function getCaptureRebindMetadata(code, actionId) {
    const keyCode = String(code || '');
    const currentActionId = String(actionId || '');
    if (keyCode === 'Escape') {
      return {
        mode: 'cancel',
        keyCode,
        actionId: '',
        rebindingAction: '',
        toastMessage: ''
      };
    }
    if (!keyCode) {
      return {
        mode: 'missing',
        keyCode,
        actionId: currentActionId,
        rebindingAction: currentActionId,
        toastMessage: 'Choose a key.'
      };
    }
    return {
      mode: 'assign',
      keyCode,
      actionId: currentActionId,
      rebindingAction: '',
      toastMessage: ''
    };
  }

  function getCaptureRebindStateAction(metadata) {
    const currentMetadata = metadata || {};
    const shouldSetRebindingAction = currentMetadata.mode === 'cancel' || currentMetadata.mode === 'assign';
    return {
      shouldSetRebindingAction,
      rebindingAction: shouldSetRebindingAction ? String(currentMetadata.rebindingAction || '') : ''
    };
  }

  function getRestoreDefaultKeybindsMetadata(options) {
    const settings = options || {};
    return {
      keybinds: createDefaultKeybinds(),
      selectedBindActionId: '',
      selectedBindSourceCode: '',
      rebindingAction: '',
      shouldClearSelectedBindAction: true,
      shouldSave: true,
      toastMessage: settings.silent ? '' : 'Default keybinds restored.'
    };
  }

  function getRestoreDefaultKeybindsStateAction(metadata) {
    const currentMetadata = metadata || {};
    const shouldClearSelectedBindAction = !!currentMetadata.shouldClearSelectedBindAction;
    return {
      shouldSetKeybinds: !!currentMetadata.keybinds,
      keybinds: currentMetadata.keybinds || null,
      shouldClearSelectedBindAction,
      shouldSetSelectedBindActionId: !shouldClearSelectedBindAction,
      selectedBindActionId: String(currentMetadata.selectedBindActionId || ''),
      shouldSetSelectedBindSourceCode: !shouldClearSelectedBindAction,
      selectedBindSourceCode: String(currentMetadata.selectedBindSourceCode || ''),
      shouldSetRebindingAction: !shouldClearSelectedBindAction,
      rebindingAction: String(currentMetadata.rebindingAction || '')
    };
  }

  function getKeybindCanvasRegionAction(region) {
    const source = region || {};
    if (source.type === 'capture-bind') return { handled: true, type: 'startRebind', actionId: source.actionId };
    if (source.type === 'reset-keybinds') return { handled: true, type: 'restoreDefaults' };
    if (source.type === 'disabled-key') return { handled: true, type: 'disabledKey', code: source.code };
    if (source.type === 'key-target') return { handled: true, type: 'keyTarget', code: source.code, ownerId: source.ownerId };
    return { handled: false, type: '' };
  }

  function getResetSkillKeybindsMetadata(bindings, state, options) {
    const settings = options || {};
    const currentBindings = bindings || {};
    const currentState = state || {};
    const before = Object.keys(currentBindings).filter((id) => id.startsWith(SKILL_BIND_PREFIX)).length;
    const selectedBindActionId = String(currentState.selectedBindActionId || '');
    const rebindingAction = String(currentState.rebindingAction || '');
    const shouldClearSelectedBindAction = !!selectedBindActionId && selectedBindActionId.startsWith(SKILL_BIND_PREFIX);
    return {
      before,
      keybinds: removeSkillKeybinds(bindings),
      rebindingAction: shouldClearSelectedBindAction || rebindingAction.startsWith(SKILL_BIND_PREFIX) ? '' : rebindingAction,
      shouldClearSelectedBindAction,
      shouldSave: !!(before || settings.forceSave),
      toastMessage: !settings.silent && before ? 'Skill keybinds reset for the new character.' : ''
    };
  }

  function getResetSkillKeybindsStateAction(metadata) {
    const currentMetadata = metadata || {};
    const shouldClearSelectedBindAction = !!currentMetadata.shouldClearSelectedBindAction;
    return {
      shouldSetKeybinds: !!currentMetadata.keybinds,
      keybinds: currentMetadata.keybinds || null,
      shouldClearSelectedBindAction,
      shouldSetRebindingAction: !shouldClearSelectedBindAction,
      rebindingAction: String(currentMetadata.rebindingAction || '')
    };
  }

  function getApplyLoadedSaveKeybindsMetadata(data) {
    const source = data || {};
    const loadedKeybinds = source.keybinds;
    const handled = !!loadedKeybinds && typeof loadedKeybinds === 'object';
    return {
      handled,
      keybinds: handled ? normalizeKeybinds(loadedKeybinds) : null,
      rebindingAction: '',
      selectedBindActionId: '',
      selectedBindSourceCode: '',
      draggingBindActionId: '',
      draggingBindSourceCode: '',
      shouldSave: handled,
      shouldRefresh: handled
    };
  }

  function getApplyLoadedSaveKeybindsStateAction(metadata) {
    const currentMetadata = metadata || {};
    const handled = !!currentMetadata.handled;
    return {
      shouldSetKeybinds: handled && !!currentMetadata.keybinds,
      keybinds: currentMetadata.keybinds || null,
      shouldSetRebindingAction: handled,
      rebindingAction: String(currentMetadata.rebindingAction || ''),
      shouldSetSelectedBindActionId: handled,
      selectedBindActionId: String(currentMetadata.selectedBindActionId || ''),
      shouldSetSelectedBindSourceCode: handled,
      selectedBindSourceCode: String(currentMetadata.selectedBindSourceCode || ''),
      shouldSetDraggingBindActionId: handled,
      draggingBindActionId: String(currentMetadata.draggingBindActionId || ''),
      shouldSetDraggingBindSourceCode: handled,
      draggingBindSourceCode: String(currentMetadata.draggingBindSourceCode || '')
    };
  }

  function isAssignableKeyCode(code) {
    const value = String(code || '').trim();
    return !!value && !BLOCKED_BINDING_KEY_SET.has(value);
  }

  function createDefaultKeybinds() {
    const used = new Set();
    const bindings = KEYBIND_ACTIONS.reduce((nextBindings, action) => {
      const key = (action.defaultKeys || []).find((code) => isAssignableKeyCode(code) && !used.has(code));
      nextBindings[action.id] = key ? [key] : [];
      if (key) used.add(key);
      return nextBindings;
    }, {});
    if (isAssignableKeyCode(ATTACK_HOLD_DEFAULT_KEY) && !used.has(ATTACK_HOLD_DEFAULT_KEY)) {
      bindings.attack = (bindings.attack || []).concat(ATTACK_HOLD_DEFAULT_KEY);
    }
    return bindings;
  }

  function normalizeKeyList(value, fallback, used) {
    const list = (Array.isArray(value) ? value : [value]).concat(fallback || []);
    const key = list
      .map((key) => String(key || '').trim())
      .find((code) => isAssignableKeyCode(code) && !(used && used.has(code)));
    if (key && used) used.add(key);
    return key ? [key] : [];
  }

  function normalizeLegacyKeybindMenuKeys(value) {
    const list = (Array.isArray(value) ? value : [value])
      .map((key) => String(key || '').trim())
      .filter(Boolean);
    return list.length === 1 && list[0] === 'Slash' ? ['Backslash'] : list;
  }

  function normalizeKeybinds(value) {
    const defaults = createDefaultKeybinds();
    if (!value || typeof value !== 'object') return defaults;
    const used = new Set();
    KEYBIND_ACTIONS.forEach((action) => {
      const source = action.id === 'keybinds' ? normalizeLegacyKeybindMenuKeys(value[action.id]) : value[action.id];
      defaults[action.id] = normalizeKeyList(source, action.defaultKeys, used);
    });
    Object.keys(value).forEach((id) => {
      if (!id.startsWith(SKILL_BIND_PREFIX)) return;
      defaults[id] = normalizeKeyList(value[id], [], used);
    });
    Object.keys(value).forEach((id) => {
      if (!id.startsWith(ITEM_BIND_PREFIX)) return;
      defaults[id] = normalizeKeyList(value[id], [], used);
    });
    if ((defaults.attack || []).length === 1 &&
      defaults.attack[0] === 'KeyJ' &&
      isAssignableKeyCode(ATTACK_HOLD_DEFAULT_KEY) &&
      !used.has(ATTACK_HOLD_DEFAULT_KEY)) {
      defaults.attack.push(ATTACK_HOLD_DEFAULT_KEY);
    }
    return defaults;
  }

  function removeSkillKeybinds(bindings) {
    const next = Object.assign({}, bindings || createDefaultKeybinds());
    Object.keys(next).forEach((id) => {
      if (id.startsWith(SKILL_BIND_PREFIX)) delete next[id];
    });
    return normalizeKeybinds(next);
  }

  function createKeybindingNormalizationUiHelpers() {
    return Object.freeze({
      formatKeyCode,
      isAssignableKeyCode,
      createDefaultKeybinds,
      normalizeKeyList,
      normalizeLegacyKeybindMenuKeys,
      normalizeKeybinds,
      removeSkillKeybinds
    });
  }

  function createKeybindingDisplayUiHelpers() {
    return Object.freeze({
      getActionIcon,
      getCanvasActionIconMetadata,
      drawCanvasActionIcon,
      iconClass,
      renderSkillIconMarkup,
      renderActionIconMarkup,
      getBindActionMetadata,
      getVirtualKeyMetadata
    });
  }

  function createKeybindingStateUiHelpers() {
    return Object.freeze({
      getKeybindLookupCacheMetadata,
      getInvalidateKeybindLookupCacheStateAction,
      findBindingOwner,
      getKeyOwner,
      getActionsForCode,
      getRestoreDefaultKeybindsMetadata,
      getRestoreDefaultKeybindsStateAction,
      getResetSkillKeybindsMetadata,
      getResetSkillKeybindsStateAction,
      getApplyLoadedSaveKeybindsMetadata,
      getApplyLoadedSaveKeybindsStateAction
    });
  }

  function createKeybindingActionCatalogUiHelpers() {
    return Object.freeze({
      getCoreActionGroup,
      getCoreBindActions,
      getDefaultActionGroups,
      getBindableActionMetadata,
      getDefaultBindActions,
      getSkillBindActionsMetadata,
      getConsumableBindActionsMetadata,
      getBindableActionsMetadata,
      getUnassignedBindActionsMetadata,
      getBindingLabel,
      getPrimaryKeyLabel
    });
  }

  function createKeybindingInteractionUiHelpers() {
    return Object.freeze({
      getSelectBindActionMetadata,
      getSelectBindActionStateAction,
      getClearSelectedBindActionStateAction,
      isKeybindClickTarget,
      getSelectedBindOutsideClickAction,
      getKeybindResetDomAction,
      getKeybindEditDomAction,
      getKeybindAssignmentDomAction,
      isKeybindCanvasRegionType,
      getClearSelectedBindFromOutsideMetadata,
      getClearSelectedBindFromOutsideStateAction,
      getBindActionClickMetadata,
      getStartRebindMetadata,
      getStartRebindStateAction,
      getKeyTargetInteractionMetadata,
      getDisabledKeyFeedbackMetadata,
      getKeyboardRebindCaptureAction,
      getCaptureRebindMetadata,
      getCaptureRebindStateAction,
      getAssignActionToKeyMetadata,
      getAssignActionToKeyStateAction,
      getClearKeyBindingMetadata,
      getClearKeyBindingStateAction,
      getKeybindCanvasRegionAction
    });
  }

  function createKeybindingDomSelectorUiHelpers() {
    return Object.freeze({
      KEYBIND_CLICK_ATTRIBUTES,
      DOM_BOUND_ACTION_DRAG_TARGET_ATTRIBUTES,
      DOM_BOUND_ACTION_DRAG_TARGET_SELECTOR,
      DOM_BIND_ACTION_DRAG_TARGET_ATTRIBUTES,
      DOM_BIND_ACTION_DRAG_TARGET_SELECTOR
    });
  }

  function createKeybindingDragDropUiHelpers() {
    return Object.freeze({
      getDomBindDragStartMetadata,
      getDomBindDragStartTargetAction,
      getDomBindDragStartTransferWriteAction,
      getDomBindDragStartStateAction,
      getDomBindDragEndMetadata,
      getDomBindDragEndStateAction,
      getDomBindOutsideDropMetadata,
      getDomBindOutsideDropStateAction,
      getDomBindDropTransferReadAction,
      getDomBindDropActionMetadata,
      getDomBindDropDispatchAction,
      getDomBindDropDispatchStateAction,
      getCanvasBindDragDropMetadata,
      getCanvasBindDragDropStateAction,
      getCanvasBindDragClickMetadata,
      getCanvasBindDragClickStateAction,
      getCanvasBindDropPointerAction,
      getCanvasBindDragStartMetadata,
      getCanvasBindDragMoveMetadata,
      getCanvasBindDragPreviewMetadata
    });
  }

  function createKeybindingCanvasLayoutUiHelpers() {
    return Object.freeze({
      getCanvasBindActionGridMetadata,
      getCanvasKeyboardLayoutMetadata
    });
  }

  const api = {
    KEYBIND_STORAGE_KEY,
    FIXED_MOVEMENT_KEYS,
    FIXED_MOVEMENT_KEY_CODES,
    createFixedMovementKeyCodes,
    KEYBIND_ACTIONS,
    ACTION_BY_ID,
    createActionById,
    DEFAULT_TRAY_ACTION_IDS,
    createDefaultTrayActionIds,
    KEYBIND_CORE_ACTION_GROUPS,
    SKILL_BIND_PREFIX,
    ITEM_BIND_PREFIX,
    ATTACK_HOLD_DEFAULT_KEY,
    BLOCKED_BINDING_KEYS,
    createBlockedBindingKeys,
    BLOCKED_BINDING_KEY_SET,
    createBlockedBindingKeySet,
    KEYBIND_CLICK_ATTRIBUTES,
    KEYBIND_CANVAS_REGION_TYPES,
    KEY_LABELS,
    KEYBOARD_ROWS,
    ACTION_ICONS,
    getActionIcon,
    getCanvasActionIconMetadata,
    drawCanvasActionIcon,
    iconClass,
    renderSkillIconMarkup,
    renderActionIconMarkup,
    formatKeyCode,
    getBindActionMetadata,
    getVirtualKeyMetadata,
    getKeybindLookupCacheMetadata,
    getInvalidateKeybindLookupCacheStateAction,
    getCoreActionGroup,
    getCoreBindActions,
    getDefaultActionGroups,
    getBindableActionMetadata,
    getDefaultBindActions,
    getSkillBindActionsMetadata,
    getConsumableBindActionsMetadata,
    getBindableActionsMetadata,
    getUnassignedBindActionsMetadata,
    getSelectBindActionMetadata,
    getSelectBindActionStateAction,
    getClearSelectedBindActionStateAction,
    DOM_BOUND_ACTION_DRAG_TARGET_ATTRIBUTES,
    DOM_BOUND_ACTION_DRAG_TARGET_SELECTOR,
    DOM_BIND_ACTION_DRAG_TARGET_ATTRIBUTES,
    DOM_BIND_ACTION_DRAG_TARGET_SELECTOR,
    isKeybindClickTarget,
    getSelectedBindOutsideClickAction,
    getKeybindResetDomAction,
    getKeybindEditDomAction,
    getKeybindAssignmentDomAction,
    isKeybindCanvasRegionType,
    getClearSelectedBindFromOutsideMetadata,
    getClearSelectedBindFromOutsideStateAction,
    getBindActionClickMetadata,
    getDomBindDragStartMetadata,
    getDomBindDragStartTargetAction,
    getDomBindDragStartTransferWriteAction,
    getDomBindDragStartStateAction,
    getDomBindDragEndMetadata,
    getDomBindDragEndStateAction,
    getDomBindOutsideDropMetadata,
    getDomBindOutsideDropStateAction,
    getDomBindDropTransferReadAction,
    getDomBindDropActionMetadata,
    getDomBindDropDispatchAction,
    getDomBindDropDispatchStateAction,
    getCanvasBindDragDropMetadata,
    getCanvasBindDragDropStateAction,
    getCanvasBindDragClickMetadata,
    getCanvasBindDragClickStateAction,
    getCanvasBindDropPointerAction,
    getCanvasBindDragStartMetadata,
    getCanvasBindDragMoveMetadata,
    getCanvasBindDragPreviewMetadata,
    getCanvasBindActionGridMetadata,
    getCanvasKeyboardLayoutMetadata,
    getAssignActionToKeyMetadata,
    getAssignActionToKeyStateAction,
    getClearKeyBindingMetadata,
    getClearKeyBindingStateAction,
    getBindingLabel,
    getPrimaryKeyLabel,
    getKeyOwner,
    getActionsForCode,
    findBindingOwner,
    getStartRebindMetadata,
    getStartRebindStateAction,
    getKeyTargetInteractionMetadata,
    getDisabledKeyFeedbackMetadata,
    getKeyboardRebindCaptureAction,
    getCaptureRebindMetadata,
    getCaptureRebindStateAction,
    getRestoreDefaultKeybindsMetadata,
    getRestoreDefaultKeybindsStateAction,
    getKeybindCanvasRegionAction,
    getResetSkillKeybindsMetadata,
    getResetSkillKeybindsStateAction,
    getApplyLoadedSaveKeybindsMetadata,
    getApplyLoadedSaveKeybindsStateAction,
    isAssignableKeyCode,
    createDefaultKeybinds,
    normalizeKeyList,
    normalizeLegacyKeybindMenuKeys,
    normalizeKeybinds,
    removeSkillKeybinds,
    createKeybindingDisplayUiHelpers,
    createKeybindingStateUiHelpers,
    createKeybindingActionCatalogUiHelpers,
    createKeybindingInteractionUiHelpers,
    createKeybindingDomSelectorUiHelpers,
    createKeybindingDragDropUiHelpers,
    createKeybindingCanvasLayoutUiHelpers,
    createKeybindingNormalizationUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.keybindings = Object.assign({}, modules.keybindings || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
