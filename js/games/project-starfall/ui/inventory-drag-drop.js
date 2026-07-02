(function initProjectStarfallUiInventoryDragDrop(global) {
  'use strict';

  const UiModules = global.ProjectStarfallUiModules || {};
  const UiInventoryConfig = (typeof require === 'function' ? require('./inventory-config.js') : null) || UiModules.inventoryConfig || {};
  const DOM_DROP_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-key-target',
    'data-starfall-pet-slot',
    'data-starfall-potential-drop-zone',
    'data-starfall-upgrade-drop-zone',
    'data-starfall-inventory-slot',
    'data-starfall-storage-slot'
  ]);
  const DOM_DROP_TARGET_SELECTOR = DOM_DROP_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_INVENTORY_DRAG_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-inventory-drag-id'
  ]);
  const DOM_INVENTORY_DRAG_TARGET_SELECTOR = DOM_INVENTORY_DRAG_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_GEAR_DRAG_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-gear-drag'
  ]);
  const DOM_GEAR_DRAG_TARGET_SELECTOR = DOM_GEAR_DRAG_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_ITEM_MENU_ACTION_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-item-menu-action'
  ]);
  const DOM_ITEM_MENU_ACTION_TARGET_SELECTOR = DOM_ITEM_MENU_ACTION_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const DOM_ITEM_CONTEXT_MENU_ROOT_ATTRIBUTES = Object.freeze([
    'data-starfall-item-context-menu-root'
  ]);
  const DOM_ITEM_CONTEXT_MENU_ROOT_SELECTOR = DOM_ITEM_CONTEXT_MENU_ROOT_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function createInventoryDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_DROP_TARGET_ATTRIBUTES,
      DOM_DROP_TARGET_SELECTOR,
      DOM_INVENTORY_DRAG_TARGET_ATTRIBUTES,
      DOM_INVENTORY_DRAG_TARGET_SELECTOR,
      DOM_GEAR_DRAG_TARGET_ATTRIBUTES,
      DOM_GEAR_DRAG_TARGET_SELECTOR,
      DOM_ITEM_MENU_ACTION_TARGET_ATTRIBUTES,
      DOM_ITEM_MENU_ACTION_TARGET_SELECTOR,
      DOM_ITEM_CONTEXT_MENU_ROOT_ATTRIBUTES,
      DOM_ITEM_CONTEXT_MENU_ROOT_SELECTOR
    });
  }

  function normalizeInventoryTabFallback(value) {
    const id = String(value || '').trim();
    return ['equipment', 'usable', 'etc', 'cards'].includes(id) ? id : 'equipment';
  }

  function getNormalizeInventoryTab(options) {
    return options && typeof options.normalizeInventoryTab === 'function'
      ? options.normalizeInventoryTab
      : UiInventoryConfig.normalizeInventoryTab || normalizeInventoryTabFallback;
  }

  function normalizeInventoryDragSource(source) {
    const id = String(source || '').trim();
    if (id === 'equipped') return 'equipped';
    if (id === 'storage') return 'storage';
    return 'inventory';
  }

  function getInventoryDragPayloadFromElement(element, options) {
    if (!element || !element.getAttribute) return null;
    const normalizeInventoryTab = getNormalizeInventoryTab(options);
    const tab = normalizeInventoryTab(element.getAttribute('data-starfall-inventory-drag-tab'));
    const id = String(element.getAttribute('data-starfall-inventory-drag-id') || '');
    if (!id) return null;
    return {
      tab,
      id,
      index: Math.max(0, Math.floor(Number(element.getAttribute('data-starfall-inventory-drag-index') || 0) || 0)),
      source: normalizeInventoryDragSource(element.getAttribute('data-starfall-inventory-drag-source'))
    };
  }

  function getInventoryContextPayloadFromDomTarget(target, options) {
    const settings = options || {};
    const root = settings.root || null;
    const getPayload = typeof settings.getInventoryDragPayloadFromElement === 'function'
      ? settings.getInventoryDragPayloadFromElement
      : getInventoryDragPayloadFromElement;
    const selector = settings.selector || DOM_INVENTORY_DRAG_TARGET_SELECTOR;
    if (!target || typeof target.closest !== 'function') return null;
    const element = target.closest(selector);
    if (!element || !root || typeof root.contains !== 'function' || !root.contains(element)) return null;
    return getPayload(element);
  }

  function getCanvasInventoryDragPayload(region) {
    if (!region) return null;
    const slotIndex = Math.max(0, Math.floor(Number(region.slotIndex || 0) || 0));
    if ((region.type === 'inventory-item' || region.type === 'equip-item') && region.uid) {
      return {
        tab: 'equipment',
        id: String(region.uid),
        index: slotIndex,
        source: 'inventory'
      };
    }
    if (region.type === 'equipment-item' && region.uid) {
      return {
        tab: 'equipment',
        id: String(region.uid),
        index: 0,
        source: 'equipped'
      };
    }
    if (region.type === 'storage-item' && region.uid) {
      return {
        tab: 'equipment',
        id: String(region.uid),
        index: slotIndex,
        source: 'storage'
      };
    }
    if ((region.type === 'consumable-item' || region.type === 'use-consumable') && region.itemId) {
      return {
        tab: 'usable',
        id: String(region.itemId),
        index: slotIndex,
        source: 'inventory'
      };
    }
    if (region.type === 'storage-consumable-item' && region.itemId) {
      return {
        tab: 'usable',
        id: String(region.itemId),
        index: slotIndex,
        source: 'storage'
      };
    }
    if (region.type === 'etc-item' && region.materialId) {
      return {
        tab: 'etc',
        id: String(region.materialId),
        index: slotIndex,
        source: 'inventory'
      };
    }
    if (region.type === 'storage-etc-item' && region.materialId) {
      return {
        tab: 'etc',
        id: String(region.materialId),
        index: slotIndex,
        source: 'storage'
      };
    }
    if ((region.type === 'card-item' || region.type === 'card-equip') && region.uid) {
      return {
        tab: 'cards',
        id: String(region.uid),
        index: slotIndex,
        source: 'inventory'
      };
    }
    return null;
  }

  function getInventoryContextPayloadFromCanvasPoint(point, options) {
    const settings = options || {};
    const getPayload = typeof settings.getCanvasInventoryDragPayload === 'function'
      ? settings.getCanvasInventoryDragPayload
      : getCanvasInventoryDragPayload;
    const findRegion = typeof settings.findCanvasRegion === 'function'
      ? settings.findCanvasRegion
      : function findCanvasRegionFallback() { return null; };
    const region = findRegion(point, (item) => !!getPayload(item));
    return region ? getPayload(region) : null;
  }

  function getMaterialContextItem(payload, options) {
    const getMaterialItemById = options && typeof options.getMaterialItemById === 'function'
      ? options.getMaterialItemById
      : function getMaterialItemByIdFallback() { return null; };
    if (!payload) return null;
    return getMaterialItemById(payload.id, payload.source === 'storage' ? 'storage' : 'inventory');
  }

  function isStorageContextAvailable(state) {
    const context = state || {};
    return context.activePanel === 'storage' || (context.openWindows || []).includes('storage');
  }

  function getItemContextMenuContext(payload, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const normalizeSource = typeof settings.normalizeInventoryDragSource === 'function'
      ? settings.normalizeInventoryDragSource
      : normalizeInventoryDragSource;
    const getSharedStorageItemByUid = settings.getSharedStorageItemByUid || function getSharedStorageItemByUidFallback() { return null; };
    const getGearItemByUid = settings.getGearItemByUid || function getGearItemByUidFallback() { return null; };
    const getSlotMeta = settings.getSlotMeta || function getSlotMetaFallback() { return {}; };
    const getConsumableItem = settings.getConsumableItem || function getConsumableItemFallback() { return null; };
    const getStorageTransferMax = settings.getStorageTransferMax || function getStorageTransferMaxFallback() { return 0; };
    const getMaterialItem = settings.getMaterialContextItem || function getMaterialContextItemFallback(candidate) {
      return getMaterialContextItem(candidate, settings);
    };
    const getCardSnapshot = settings.getCardSnapshot || function getCardSnapshotFallback() { return { inventory: [] }; };
    const getCardName = settings.getCardName || function getCardNameFallback(card) { return card && card.name ? card.name : 'Card'; };
    const formatInteger = settings.formatAbbreviatedInteger || function formatAbbreviatedIntegerFallback(value) { return String(value); };
    if (!payload) return null;
    const tab = normalizeInventoryTab(payload.tab);
    const source = normalizeSource(payload.source);
    const id = String(payload.id || '');
    if (!id) return null;
    if (tab === 'equipment') {
      const item = source === 'storage' ? getSharedStorageItemByUid(id) : getGearItemByUid(id);
      if (!item) return null;
      const slotMeta = getSlotMeta(item.slot) || {};
      return {
        kind: 'equipment',
        tab,
        source,
        id,
        item,
        title: item.name || 'Equipment',
        subtitle: `${source === 'equipped' ? 'Equipped' : source === 'storage' ? 'Storage' : 'Inventory'} ${slotMeta.label || 'Gear'}`
      };
    }
    if (tab === 'usable') {
      const item = getConsumableItem(id);
      if (!item) return null;
      const count = getStorageTransferMax({ tab, id, source });
      if (count <= 0) return null;
      return {
        kind: 'consumable',
        tab,
        source,
        id,
        item,
        count,
        title: item.name || 'Usable Item',
        subtitle: `${source === 'storage' ? 'Storage' : 'Inventory'} usable x${formatInteger(count)}`
      };
    }
    if (tab === 'etc') {
      const item = getMaterialItem({ tab, id, source });
      if (!item) return null;
      const count = getStorageTransferMax({ tab, id, source });
      if (count <= 0) return null;
      return {
        kind: 'material',
        tab,
        source,
        id,
        item,
        count,
        title: item.name || 'Etc Item',
        subtitle: `${source === 'storage' ? 'Storage' : 'Inventory'} etc x${formatInteger(count)}`
      };
    }
    if (tab === 'cards') {
      const snapshot = getCardSnapshot() || {};
      const card = (snapshot.inventory || []).find((entry) => entry && entry.uid === id);
      if (!card || source !== 'inventory') return null;
      return {
        kind: 'card',
        tab,
        source,
        id,
        item: card,
        title: getCardName(card),
        subtitle: `Card tier ${Number(card.rank || 1)} x${formatInteger(card.quantity || 1)}`
      };
    }
    return null;
  }

  function createInventoryPayloadContextUiHelpers() {
    return Object.freeze({
      normalizeInventoryDragSource,
      getInventoryDragPayloadFromElement,
      getInventoryContextPayloadFromDomTarget,
      getCanvasInventoryDragPayload,
      getInventoryContextPayloadFromCanvasPoint,
      getMaterialContextItem,
      isStorageContextAvailable,
      getItemContextMenuContext
    });
  }

  function getStoreItemContextPayloadAction(payload, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const getStorageTransferMax = settings.getStorageTransferMax || function getStorageTransferMaxFallback() { return 0; };
    const getFirstSharedStorageSlotIndex = settings.getFirstSharedStorageSlotIndex || function getFirstSharedStorageSlotIndexFallback() { return 0; };
    if (!payload || payload.source === 'storage') return { handled: false, type: '', payload: payload || null };
    const tab = normalizeInventoryTab(payload.tab);
    const max = getStorageTransferMax(payload);
    const targetIndex = getFirstSharedStorageSlotIndex(tab);
    if (max > 1) {
      return {
        handled: true,
        type: 'openStorageTransferQuantityPrompt',
        payload,
        target: 'storage',
        targetIndex,
        max
      };
    }
    return {
      handled: true,
      type: 'transferInventoryPayloadToStorage',
      payload,
      target: 'storage',
      targetIndex,
      quantity: 1,
      max
    };
  }

  function getWithdrawItemContextPayloadAction(payload, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const getStorageTransferMax = settings.getStorageTransferMax || function getStorageTransferMaxFallback() { return 0; };
    const getFirstInventorySlotIndex = settings.getFirstInventorySlotIndex || function getFirstInventorySlotIndexFallback() { return 0; };
    if (!payload || payload.source !== 'storage') return { handled: false, type: '', payload: payload || null };
    const tab = normalizeInventoryTab(payload.tab);
    const max = getStorageTransferMax(payload);
    const targetIndex = getFirstInventorySlotIndex(tab);
    if (max > 1) {
      return {
        handled: true,
        type: 'openStorageTransferQuantityPrompt',
        payload,
        target: 'inventory',
        targetIndex,
        max
      };
    }
    return {
      handled: true,
      type: 'transferStoragePayloadToInventory',
      payload,
      target: 'inventory',
      targetIndex,
      quantity: 1,
      max
    };
  }

  function getDropItemContextPayloadAction(payload, options) {
    const settings = options || {};
    const getDropCandidate = settings.getDropCandidateForInventoryPayload || function getDropCandidateForInventoryPayloadFallback() { return null; };
    const getQuantityMax = settings.getDropQuantityMax || getDropQuantityMax;
    const candidate = getDropCandidate(payload);
    if (!candidate) {
      return {
        handled: false,
        type: '',
        payload: payload || null,
        candidate: null,
        worldPoint: null
      };
    }
    const max = getQuantityMax(candidate);
    const worldPoint = null;
    if (max > 1) {
      return {
        handled: true,
        type: 'openDropQuantityPrompt',
        payload,
        candidate,
        max,
        worldPoint
      };
    }
    return {
      handled: true,
      type: 'dropInventoryCandidateAtWorldPoint',
      payload,
      candidate,
      quantity: 1,
      max,
      worldPoint
    };
  }

  function createInventoryContextPayloadActionUiHelpers() {
    return Object.freeze({
      getStoreItemContextPayloadAction,
      getWithdrawItemContextPayloadAction,
      getDropItemContextPayloadAction
    });
  }

  function getStorageSlotPayloadDropAction(payload, tab, targetIndex, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const getStorageTransferMax = settings.getStorageTransferMax || function getStorageTransferMaxFallback() { return 0; };
    if (!payload) return { handled: false, type: '', payload: null };
    const normalizedTab = normalizeInventoryTab(tab);
    const slotIndex = Math.max(0, Math.floor(Number(targetIndex || 0) || 0));
    if (payload.tab !== normalizedTab) {
      return {
        handled: true,
        type: 'showToast',
        payload,
        tab: normalizedTab,
        targetIndex: slotIndex,
        message: 'Drag items into the matching storage tab.'
      };
    }
    if (payload.source === 'storage') {
      return {
        handled: true,
        type: 'moveSharedStorageSlot',
        payload,
        tab: normalizedTab,
        fromIndex: Math.max(0, Math.floor(Number(payload.index || 0) || 0)),
        targetIndex: slotIndex
      };
    }
    const max = getStorageTransferMax(payload);
    if (max > 1) {
      return {
        handled: true,
        type: 'openStorageTransferQuantityPrompt',
        payload,
        target: 'storage',
        targetIndex: slotIndex,
        max
      };
    }
    return {
      handled: true,
      type: 'transferInventoryPayloadToStorage',
      payload,
      target: 'storage',
      targetIndex: slotIndex,
      quantity: 1,
      max
    };
  }

  function getInventorySlotPayloadDropAction(payload, tab, targetIndex, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const getUnlockedCapacity = settings.getInventoryUnlockedCapacity || function getInventoryUnlockedCapacityFallback() { return 0; };
    const getStorageTransferMax = settings.getStorageTransferMax || function getStorageTransferMaxFallback() { return 0; };
    if (!payload) return { handled: false, type: '', payload: null };
    const normalizedTab = normalizeInventoryTab(tab);
    const slotIndex = Math.max(0, Math.floor(Number(targetIndex || 0) || 0));
    if (slotIndex >= getUnlockedCapacity(normalizedTab)) {
      return {
        handled: true,
        type: 'showToast',
        payload,
        tab: normalizedTab,
        targetIndex: slotIndex,
        message: 'Unlock this inventory section first.'
      };
    }
    if (payload.tab !== normalizedTab) {
      return {
        handled: true,
        type: 'showToast',
        payload,
        tab: normalizedTab,
        targetIndex: slotIndex,
        message: 'Drag items within the same inventory tab.'
      };
    }
    if (payload.source === 'storage') {
      const max = getStorageTransferMax(payload);
      if (max > 1) {
        return {
          handled: true,
          type: 'openStorageTransferQuantityPrompt',
          payload,
          target: 'inventory',
          targetIndex: slotIndex,
          max
        };
      }
      return {
        handled: true,
        type: 'transferStoragePayloadToInventory',
        payload,
        target: 'inventory',
        targetIndex: slotIndex,
        quantity: 1,
        max
      };
    }
    if (payload.source === 'equipped') {
      if (normalizedTab !== 'equipment' || !settings.canUnequipItem) {
        return {
          handled: true,
          type: 'showToast',
          payload,
          tab: normalizedTab,
          targetIndex: slotIndex,
          message: 'Drag equipped gear into an equipment inventory slot.'
        };
      }
      return {
        handled: true,
        type: 'unequipItem',
        payload,
        tab: normalizedTab,
        id: payload.id,
        targetIndex: slotIndex,
        shouldRefresh: true
      };
    }
    return {
      handled: true,
      type: 'moveInventorySlot',
      payload,
      tab: normalizedTab,
      fromIndex: Math.max(0, Math.floor(Number(payload.index || 0) || 0)),
      targetIndex: slotIndex,
      canMove: !!settings.canMoveInventorySlot,
      shouldRefresh: true
    };
  }

  function getInventoryPayloadCanvasDropAction(payload, target, options) {
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const itemBindPrefix = String(settings.itemBindPrefix || 'item:');
    const hasCanvasBlocker = typeof settings.hasCanvasBlocker === 'function'
      ? settings.hasCanvasBlocker
      : () => !!settings.hasCanvasBlocker;
    const getDropCandidate = settings.getDropCandidateForInventoryPayload || function getDropCandidateForInventoryPayloadFallback() { return null; };
    const getQuantityMax = settings.getDropQuantityMax || getDropQuantityMax;
    if (!payload) return { handled: false, type: '', payload: null };
    const dropTarget = target || null;
    if (dropTarget && dropTarget.type === 'key-target') {
      if (payload.source === 'storage') {
        return {
          handled: true,
          type: 'showToast',
          message: 'Withdraw usable items before assigning them to keys.',
          returnValue: false
        };
      }
      if (payload.tab !== 'usable') {
        return {
          handled: true,
          type: 'showToast',
          message: 'Only usable items can be assigned to keys from inventory.',
          returnValue: false
        };
      }
      return {
        handled: true,
        type: 'assignActionToKey',
        actionId: `${itemBindPrefix}${payload.id}`,
        keyCode: dropTarget.code,
        returnValue: true
      };
    }
    if (dropTarget && (dropTarget.type === 'potential-prompt-drop-zone' || dropTarget.type === 'upgrade-prompt-drop-zone')) {
      const isUpgrade = dropTarget.type === 'upgrade-prompt-drop-zone';
      return {
        handled: true,
        type: isUpgrade ? 'setUpgradePromptTarget' : 'setPotentialPromptTarget',
        id: payload.id,
        shouldSetTarget: payload.tab === 'equipment',
        failureMessage: isUpgrade ? 'Drag gear into the upgrade prompt.' : 'Drag gear into the prism prompt.'
      };
    }
    if (dropTarget && dropTarget.type === 'inventory-slot') {
      return {
        handled: true,
        type: 'handleInventorySlotPayloadDrop',
        payload,
        tab: normalizeInventoryTab(dropTarget.tabId),
        targetIndex: dropTarget.slotIndex
      };
    }
    if (dropTarget && dropTarget.type === 'storage-slot') {
      return {
        handled: true,
        type: 'handleStorageSlotPayloadDrop',
        payload,
        tab: normalizeInventoryTab(dropTarget.tabId),
        targetIndex: dropTarget.slotIndex
      };
    }
    if (hasCanvasBlocker()) {
      return {
        handled: true,
        type: 'showToast',
        message: 'Drop items on the map or in a matching slot.',
        returnValue: false
      };
    }
    if (payload.source === 'storage') {
      return {
        handled: true,
        type: 'showToast',
        message: 'Withdraw items from storage before dropping them.',
        returnValue: false
      };
    }
    const candidate = getDropCandidate(payload);
    if (!candidate) return { handled: false, type: '', payload, candidate: null };
    const worldPoint = null;
    const max = getQuantityMax(candidate);
    if (max > 1) {
      return {
        handled: true,
        type: 'openDropQuantityPrompt',
        payload,
        candidate,
        max,
        worldPoint
      };
    }
    return {
      handled: true,
      type: 'dropInventoryCandidateAtWorldPoint',
      payload,
      candidate,
      quantity: 1,
      max,
      worldPoint
    };
  }

  function getCanvasGearDragUid(region) {
    if (!region) return '';
    return (region.type === 'inventory-item' || region.type === 'equip-item' || region.type === 'equipment-item') && region.uid
      ? String(region.uid)
      : '';
  }

  function createInventoryPayloadDropUiHelpers() {
    return Object.freeze({
      getInventorySlotPayloadDropAction,
      getInventoryPayloadCanvasDropAction,
      getCanvasGearDragUid
    });
  }

  function serializeInventoryDragPayload(payload, options) {
    const normalizeSource = options && typeof options.normalizeInventoryDragSource === 'function'
      ? options.normalizeInventoryDragSource
      : normalizeInventoryDragSource;
    return payload ? `${payload.tab}|${payload.id}|${Math.max(0, Math.floor(Number(payload.index || 0) || 0))}|${normalizeSource(payload.source)}` : '';
  }

  function parseInventoryDragPayload(value, options) {
    const normalizeInventoryTab = getNormalizeInventoryTab(options);
    const normalizeSource = options && typeof options.normalizeInventoryDragSource === 'function'
      ? options.normalizeInventoryDragSource
      : normalizeInventoryDragSource;
    const parts = String(value || '').split('|');
    if (parts.length < 2) return null;
    const tab = normalizeInventoryTab(parts[0]);
    const id = parts[1] || '';
    if (!id) return null;
    return {
      tab,
      id,
      index: Math.max(0, Math.floor(Number(parts[2] || 0) || 0)),
      source: normalizeSource(parts[3])
    };
  }

  function getInventoryDragPayloadEventReadAction(options) {
    const settings = options || {};
    return {
      shouldReadPayload: !!settings.hasDataTransfer,
      payloadDataType: 'application/x-starfall-inventory'
    };
  }

  function getInventoryDragPayloadFromEventData(rawPayload, fallbackPayload, options) {
    const parsePayload = options && typeof options.parseInventoryDragPayload === 'function'
      ? options.parseInventoryDragPayload
      : parseInventoryDragPayload;
    return parsePayload(rawPayload) || fallbackPayload || null;
  }

  function getNativeDragPreviewCleanupAction(previewElement) {
    return {
      shouldRemoveElement: !!(previewElement && previewElement.parentNode),
      element: previewElement || null,
      shouldClearPreviewElement: true,
      nextPreviewElement: null
    };
  }

  function getNativeDragPreviewSetupAction(options) {
    const settings = options || {};
    return {
      shouldCreatePreview: !!(settings.hasDocument && settings.hasBody && settings.hasTransfer && settings.canSetDragImage),
      actionId: settings.actionId || '',
      kind: settings.kind,
      label: settings.label || ''
    };
  }

  function getNativeDragLabel(source, fallback) {
    if (!source || !source.getAttribute) return fallback || 'Drag';
    const labelled = source.querySelector && source.querySelector('[aria-label]');
    const strong = source.querySelector && source.querySelector('strong');
    return source.getAttribute('aria-label') ||
      source.getAttribute('title') ||
      (labelled && labelled.getAttribute('aria-label')) ||
      (strong && strong.textContent) ||
      fallback ||
      'Drag';
  }

  function getNativeDragPreviewMetadata(options) {
    const settings = options || {};
    const kind = String(settings.kind || '');
    const labelLimit = Object.prototype.hasOwnProperty.call(settings, 'labelLimit')
      ? Math.max(0, Math.floor(Number(settings.labelLimit || 0) || 0))
      : 32;
    const minOffset = Object.prototype.hasOwnProperty.call(settings, 'minOffset')
      ? Math.max(0, Number(settings.minOffset || 0) || 0)
      : 8;
    const rect = settings.rect || {};
    return {
      fallbackSymbol: kind === 'equipment' ? 'GEAR' : kind === 'etc' ? 'ETC' : kind === 'usable' ? 'USE' : 'ACT',
      label: String(settings.label || '').slice(0, labelLimit),
      labelLimit,
      imageOffsetX: Math.max(minOffset, Number(rect.width || 0) / 2),
      imageOffsetY: Math.max(minOffset, Number(rect.height || 0) / 2),
      minOffset
    };
  }

  function getNativeDragIconCloneMetadata(source, actionId) {
    const gearUid = source && typeof source.getAttribute === 'function'
      ? String(source.getAttribute('data-starfall-gear-drag') || '')
      : '';
    const bindActionId = String(actionId || '');
    return {
      shouldCloneSourceIcon: !!(source && typeof source.querySelector === 'function'),
      iconSelector: '.project-starfall-inventory-icon-wrap, .project-starfall-action-icon, .project-starfall-skill-icon, .project-starfall-gear-icon, .project-starfall-card-icon',
      gearUid,
      shouldLookupGearItem: !!gearUid,
      gearIconClass: 'project-starfall-gear-icon',
      gearIconAriaHidden: 'true',
      gearAssetClass: 'project-starfall-item-art',
      actionId: bindActionId,
      shouldLookupBindableAction: !!bindActionId,
      actionIconClass: 'project-starfall-drag-preview-icon'
    };
  }

  function createInventoryNativeDragUiHelpers() {
    return Object.freeze({
      serializeInventoryDragPayload,
      parseInventoryDragPayload,
      getInventoryDragPayloadEventReadAction,
      getInventoryDragPayloadFromEventData,
      getNativeDragPreviewCleanupAction,
      getNativeDragPreviewSetupAction,
      getNativeDragLabel,
      getNativeDragPreviewMetadata,
      getNativeDragIconCloneMetadata
    });
  }

  function getDomInventoryDragStartAction(target, options) {
    const settings = options || {};
    const source = target && typeof target.closest === 'function'
      ? target.closest(DOM_INVENTORY_DRAG_TARGET_SELECTOR)
      : null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (!source || !hasAttribute('data-starfall-inventory-drag-id')) {
      return { handled: false, type: '', target: source || null };
    }
    const payload = getInventoryDragPayloadFromElement(source, settings);
    if (!payload) return { handled: false, type: '', target: source, payload: null };
    const itemBindPrefix = String(settings.itemBindPrefix || 'item:');
    const gearUid = payload.tab === 'equipment' ? payload.id : '';
    const actionId = payload.tab === 'usable' ? `${itemBindPrefix}${payload.id}` : '';
    return {
      handled: true,
      type: 'startInventoryDrag',
      target: source,
      payload,
      gearUid,
      actionId,
      dataTransferEffectAllowed: 'move',
      dataTransferGear: gearUid,
      dataTransferText: actionId || (gearUid ? `gear:${gearUid}` : `item:${payload.id}`),
      preview: { actionId, kind: payload.tab },
      shouldSelectAction: !!actionId,
      shouldMarkDragging: true
    };
  }

  function getDomInventoryDragStartGateAction(dragAction, options) {
    const action = dragAction || {};
    const settings = options || {};
    const hasValidTarget = !!settings.hasValidTarget;
    return {
      shouldStart: !!(action.handled && hasValidTarget),
      handled: !!action.handled,
      hasValidTarget,
      type: action.type || ''
    };
  }

  function getDomGearDragStartAction(target) {
    const source = target && typeof target.closest === 'function'
      ? target.closest(DOM_GEAR_DRAG_TARGET_SELECTOR)
      : null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (!source || !hasAttribute('data-starfall-gear-drag')) {
      return { handled: false, type: '', target: source || null };
    }
    const uid = String(getAttribute('data-starfall-gear-drag') || '');
    if (!uid) return { handled: false, type: '', target: source, uid };
    const payload = {
      tab: 'equipment',
      id: uid,
      index: 0,
      source: getAttribute('data-starfall-inventory-drag-source') === 'equipped' ? 'equipped' : 'inventory'
    };
    return {
      handled: true,
      type: 'startGearDrag',
      target: source,
      uid,
      payload,
      dataTransferEffectAllowed: 'move',
      dataTransferGear: uid,
      dataTransferText: `gear:${uid}`,
      preview: { kind: 'equipment' },
      shouldMarkDragging: true
    };
  }

  function getDomGearDragStartGateAction(dragAction, options) {
    const action = dragAction || {};
    const settings = options || {};
    const hasValidTarget = !!settings.hasValidTarget;
    const hasPromptItem = !!settings.hasPromptItem;
    return {
      shouldStart: !!(action.handled && hasValidTarget && hasPromptItem),
      handled: !!action.handled,
      hasValidTarget,
      hasPromptItem,
      uid: String(action.uid || '')
    };
  }

  function getDomInventoryDragStartStateAction(dragAction) {
    const action = dragAction || {};
    const actionId = String(action.actionId || '');
    const gearUid = String(action.gearUid || action.uid || '');
    return {
      shouldApply: !!action.handled,
      shouldSetInventoryDragPayload: !!action.payload,
      inventoryDragPayload: action.payload || null,
      shouldSetDraggingGearUid: !!gearUid,
      draggingGearUid: gearUid,
      shouldSetDraggingBindActionId: !!actionId,
      draggingBindActionId: actionId,
      shouldSelectAction: !!(action.shouldSelectAction && actionId),
      selectActionId: actionId,
      selectActionCode: '',
      shouldSetNativeDragPreview: !!action.target,
      previewTarget: action.target || null,
      preview: action.preview || {},
      shouldMarkDragging: !!action.shouldMarkDragging,
      draggingTarget: action.target || null,
      draggingClass: 'is-dragging'
    };
  }

  function getDomInventoryDragStartTransferWriteAction(dragAction, serializedPayload) {
    const action = dragAction || {};
    const shouldWriteTransfer = !!action.handled;
    return {
      shouldWriteTransfer,
      effectAllowed: action.dataTransferEffectAllowed || 'move',
      entries: [
        {
          type: 'application/x-starfall-inventory',
          value: String(serializedPayload || ''),
          shouldWrite: shouldWriteTransfer
        },
        {
          type: 'application/x-starfall-gear',
          value: String(action.dataTransferGear || ''),
          shouldWrite: shouldWriteTransfer && !!action.dataTransferGear
        },
        {
          type: 'text/plain',
          value: String(action.dataTransferText || ''),
          shouldWrite: shouldWriteTransfer
        }
      ]
    };
  }

  function getDomInventoryDragEndCleanupAction() {
    return {
      shouldClearDraggingGearUid: true,
      draggingGearUid: '',
      shouldClearInventoryDragPayload: true,
      inventoryDragPayload: null,
      shouldClearNativeDragPreview: true,
      shouldClearDraggingClass: true,
      draggingClassSelector: '.is-dragging'
    };
  }

  function getDomDropTargetAction(target) {
    const source = target && typeof target.closest === 'function'
      ? target.closest(DOM_DROP_TARGET_SELECTOR)
      : null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : (name) => getAttribute(name) !== null && getAttribute(name) !== '';
    if (!source) return { handled: false, type: '', target: null };
    const isKeyTarget = hasAttribute('data-starfall-key-target');
    const isLockedInventorySlot = hasAttribute('data-starfall-inventory-locked');
    return {
      handled: true,
      type: 'dropTarget',
      target: source,
      dropEffect: isLockedInventorySlot ? 'none' : isKeyTarget ? 'copy' : 'move',
      isInventorySlot: hasAttribute('data-starfall-inventory-slot'),
      inventorySlot: getAttribute('data-starfall-inventory-slot'),
      inventorySlotTab: getAttribute('data-starfall-inventory-slot-tab'),
      isStorageSlot: hasAttribute('data-starfall-storage-slot'),
      storageSlot: getAttribute('data-starfall-storage-slot'),
      storageSlotTab: getAttribute('data-starfall-storage-slot-tab'),
      isPotentialDropZone: hasAttribute('data-starfall-potential-drop-zone'),
      isUpgradeDropZone: hasAttribute('data-starfall-upgrade-drop-zone'),
      petSlot: getAttribute('data-starfall-pet-slot'),
      keyTarget: getAttribute('data-starfall-key-target')
    };
  }

  function getDomDragOverAction(dropTargetAction, options) {
    const settings = options || {};
    if (!settings.hasValidDropTarget) {
      const shouldAllowCanvasDrop = !!settings.isCanvasTarget && !!settings.hasInventoryPayload;
      return {
        handled: shouldAllowCanvasDrop,
        type: shouldAllowCanvasDrop ? 'canvasInventoryDrop' : '',
        shouldPreventDefault: shouldAllowCanvasDrop,
        shouldSetDropEffect: shouldAllowCanvasDrop,
        dropEffect: shouldAllowCanvasDrop ? 'move' : ''
      };
    }
    const dropEffect = dropTargetAction && dropTargetAction.dropEffect || '';
    return {
      handled: true,
      type: 'domDropTarget',
      shouldPreventDefault: true,
      shouldSetDropEffect: !!dropEffect,
      dropEffect
    };
  }

  function getDomCanvasOutsideDropAction(options) {
    const settings = options || {};
    if (!settings.isCanvasTarget) {
      return {
        handled: false,
        type: '',
        shouldPreventDefault: false,
        shouldDropInventoryPayload: false,
        shouldHandleDragEnd: false,
        shouldReturnEarly: false
      };
    }
    if (!settings.hasInventoryPayload) {
      return {
        handled: true,
        type: 'canvasDropNoPayload',
        shouldPreventDefault: false,
        shouldDropInventoryPayload: false,
        shouldHandleDragEnd: false,
        shouldReturnEarly: true
      };
    }
    return {
      handled: true,
      type: 'dropInventoryPayloadAtCanvasPoint',
      shouldPreventDefault: true,
      shouldDropInventoryPayload: true,
      shouldHandleDragEnd: true,
      shouldReturnEarly: false
    };
  }

  function getDomTargetDropStartAction(dropTargetAction, options) {
    const settings = options || {};
    const hasValidDropTarget = !!settings.hasValidDropTarget;
    return {
      handled: hasValidDropTarget,
      type: hasValidDropTarget ? 'domDropTarget' : '',
      target: hasValidDropTarget && dropTargetAction ? dropTargetAction.target : null,
      shouldPreventDefault: hasValidDropTarget,
      shouldReadInventoryPayload: hasValidDropTarget
    };
  }

  function getDomSlotPayloadDropAction(dropTargetAction, inventoryPayload, options) {
    const normalizeTab = options && typeof options.normalizeInventoryTab === 'function'
      ? options.normalizeInventoryTab
      : normalizeInventoryTab;
    const action = dropTargetAction || {};
    if (action.isInventorySlot) {
      const targetIndex = Math.max(0, Math.floor(Number(action.inventorySlot || 0) || 0));
      if (!inventoryPayload) {
        return {
          handled: true,
          type: 'showToast',
          message: 'Drag items within the same inventory tab.',
          target: 'inventory',
          tab: normalizeTab(action.inventorySlotTab),
          targetIndex,
          shouldHandleDragEnd: true
        };
      }
      return {
        handled: true,
        type: 'handleInventorySlotPayloadDrop',
        message: '',
        target: 'inventory',
        payload: inventoryPayload,
        tab: normalizeTab(action.inventorySlotTab),
        targetIndex,
        shouldHandleDragEnd: true
      };
    }
    if (action.isStorageSlot) {
      const targetIndex = Math.max(0, Math.floor(Number(action.storageSlot || 0) || 0));
      if (!inventoryPayload) {
        return {
          handled: true,
          type: 'showToast',
          message: 'Drag items into a matching storage slot.',
          target: 'storage',
          tab: normalizeTab(action.storageSlotTab),
          targetIndex,
          shouldHandleDragEnd: true
        };
      }
      return {
        handled: true,
        type: 'handleStorageSlotPayloadDrop',
        message: '',
        target: 'storage',
        payload: inventoryPayload,
        tab: normalizeTab(action.storageSlotTab),
        targetIndex,
        shouldHandleDragEnd: true
      };
    }
    return {
      handled: false,
      type: '',
      message: '',
      target: '',
      payload: null,
      tab: '',
      targetIndex: 0,
      shouldHandleDragEnd: false
    };
  }

  function getDomPromptDropTransferReadAction(dropTargetAction, inventoryPayload) {
    const action = dropTargetAction || {};
    const shouldReadPromptTransfer = !!(action.isPotentialDropZone || action.isUpgradeDropZone) &&
      !(inventoryPayload && inventoryPayload.tab === 'equipment');
    return {
      shouldReadPromptTransfer,
      shouldReadGear: shouldReadPromptTransfer,
      gearDataType: 'application/x-starfall-gear',
      shouldReadTextIfNoGear: shouldReadPromptTransfer,
      textDataType: 'text/plain'
    };
  }

  function getDomPromptDropAction(dropTargetAction, inventoryPayload, options) {
    const action = dropTargetAction || {};
    const isPotentialDropZone = !!action.isPotentialDropZone;
    const isUpgradeDropZone = !!action.isUpgradeDropZone;
    if (!isPotentialDropZone && !isUpgradeDropZone) {
      return {
        handled: false,
        type: '',
        uid: '',
        failureMessage: '',
        shouldHandleDragEnd: false
      };
    }
    const settings = options || {};
    const transferText = String(settings.dataTransferText || '');
    const uid = inventoryPayload && inventoryPayload.tab === 'equipment'
      ? inventoryPayload.id
      : String(settings.dataTransferGear || '') || transferText.replace(/^gear:/, '') || String(settings.draggingGearUid || '');
    return {
      handled: true,
      type: isUpgradeDropZone ? 'setUpgradePromptTarget' : 'setPotentialPromptTarget',
      uid: String(uid || ''),
      failureMessage: isUpgradeDropZone ? 'Drag gear into the upgrade prompt.' : 'Drag gear into the prism prompt.',
      shouldHandleDragEnd: true
    };
  }

  function createInventoryDomDragDropUiHelpers() {
    return Object.freeze({
      getDomInventoryDragStartAction,
      getDomInventoryDragStartGateAction,
      getDomGearDragStartAction,
      getDomGearDragStartGateAction,
      getDomInventoryDragStartStateAction,
      getDomInventoryDragStartTransferWriteAction,
      getDomInventoryDragEndCleanupAction,
      getDomDropTargetAction,
      getDomDragOverAction,
      getDomCanvasOutsideDropAction,
      getDomTargetDropStartAction,
      getDomSlotPayloadDropAction,
      getDomPromptDropTransferReadAction,
      getDomPromptDropAction
    });
  }

  function getDropCandidateForEquipment(uid, item, source, options) {
    const id = String(uid || '');
    const normalizeSource = options && typeof options.normalizeInventoryDragSource === 'function'
      ? options.normalizeInventoryDragSource
      : normalizeInventoryDragSource;
    if (!item) return null;
    return {
      type: 'equipment',
      id,
      name: item.name || 'item',
      max: 1,
      source: normalizeSource(source)
    };
  }

  function getDropCandidateForConsumable(itemId, item, count) {
    const id = String(itemId || '');
    const quantity = Math.max(0, Math.floor(Number(count || 0) || 0));
    if (!item || !quantity) return null;
    return {
      type: 'consumable',
      id,
      name: item.name || id,
      max: quantity
    };
  }

  function getDropCandidateForMaterial(materialId, item) {
    const id = String(materialId || '');
    if (!item) return null;
    return {
      type: 'material',
      id,
      name: item.name || id,
      max: Math.max(1, Math.floor(Number(item.count || 1) || 1))
    };
  }

  function getDropCandidateForCanvasRegion(region, options) {
    const settings = options || {};
    const getEquipmentCandidate = settings.getDropCandidateForEquipment || function getDropCandidateForEquipmentFallback() { return null; };
    const getConsumableCandidate = settings.getDropCandidateForConsumable || function getDropCandidateForConsumableFallback() { return null; };
    const getMaterialCandidate = settings.getDropCandidateForMaterial || function getDropCandidateForMaterialFallback() { return null; };
    if (!region) return null;
    if ((region.type === 'inventory-item' || region.type === 'equip-item' || region.type === 'equipment-item') && region.uid) return getEquipmentCandidate(region.uid);
    if ((region.type === 'consumable-item' || region.type === 'use-consumable') && region.itemId) return getConsumableCandidate(region.itemId);
    if (region.type === 'etc-item' && region.materialId) return getMaterialCandidate(region.materialId);
    return null;
  }

  function getDropCandidateForInventoryPayload(payload, options) {
    const settings = options || {};
    const getEquipmentCandidate = settings.getDropCandidateForEquipment || function getDropCandidateForEquipmentFallback() { return null; };
    const getConsumableCandidate = settings.getDropCandidateForConsumable || function getDropCandidateForConsumableFallback() { return null; };
    const getMaterialCandidate = settings.getDropCandidateForMaterial || function getDropCandidateForMaterialFallback() { return null; };
    if (!payload) return null;
    if (payload.source === 'storage') return null;
    if (payload.tab === 'equipment') return getEquipmentCandidate(payload.id, payload.source);
    if (payload.tab === 'usable') return getConsumableCandidate(payload.id);
    if (payload.tab === 'etc') return getMaterialCandidate(payload.id);
    return null;
  }

  function getDropQuantityMax(candidate) {
    return Math.max(1, Math.floor(Number(candidate && candidate.max || 1) || 1));
  }

  function normalizeDropPromptQuantity(quantity, max, options) {
    const clamp = options && typeof options.clamp === 'function'
      ? options.clamp
      : (value, min, high) => Math.max(min, Math.min(high, value));
    return clamp(Math.floor(Number(quantity) || 0), 1, getDropQuantityMax({ max }));
  }

  function createDropQuantityPrompt(candidate, worldPoint) {
    const max = getDropQuantityMax(candidate);
    if (!candidate || max <= 1) return null;
    return {
      candidate: Object.assign({}, candidate),
      max,
      quantity: max,
      worldPoint: worldPoint ? Object.assign({}, worldPoint) : null
    };
  }

  function isPendingInventoryDrop(pending, type, id) {
    return !!(pending && pending.type === type && pending.id === String(id || ''));
  }

  function createInventoryDropCandidateUiHelpers() {
    return Object.freeze({
      getDropCandidateForEquipment,
      getDropCandidateForConsumable,
      getDropCandidateForMaterial,
      getDropCandidateForCanvasRegion,
      getDropCandidateForInventoryPayload,
      getDropQuantityMax,
      normalizeDropPromptQuantity,
      createDropQuantityPrompt,
      isPendingInventoryDrop
    });
  }

  function getCanvasInventoryActivationRegion(region, point, options) {
    if (!region) return null;
    if ((region.type === 'inventory-item' || region.type === 'equip-item') && region.uid) {
      return { type: 'equipment', id: String(region.uid) };
    }
    if (region.type === 'equipment-item' && region.uid) {
      return { type: 'equipped-equipment', id: String(region.uid) };
    }
    if ((region.type === 'consumable-item' || region.type === 'use-consumable') && region.itemId) {
      return { type: 'consumable', id: String(region.itemId) };
    }
    if (region.type === 'etc-item' && region.materialId) {
      return { type: 'etc', id: String(region.materialId) };
    }
    const settings = options || {};
    if (region.type === 'bind-action' && point && typeof settings.findConsumableRegionAtPoint === 'function') {
      const consumableRegion = settings.findConsumableRegionAtPoint(point);
      if (consumableRegion && consumableRegion.itemId) return { type: 'consumable', id: String(consumableRegion.itemId) };
    }
    return null;
  }

  function getInventoryActivationKey(activation) {
    return activation ? `${activation.type}:${activation.id}` : '';
  }

  function isCanvasInventoryDoubleClick(activation, point, lastClick, now, options) {
    const last = lastClick || null;
    const currentPoint = point || null;
    if (!last || !activation || !currentPoint) return false;
    if (last.key !== getInventoryActivationKey(activation)) return false;
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

  function getCanvasInventoryActivationClickAction(activation, point, isDoubleClick, now) {
    if (!activation) {
      return {
        handled: false,
        type: '',
        activation: null,
        click: null,
        shouldClearCanvasInventoryClick: false,
        shouldActivateInventory: false,
        shouldRememberCanvasInventoryClick: false,
        shouldPreventDefault: false,
        shouldReturnEarly: false
      };
    }
    if (isDoubleClick) {
      return {
        handled: true,
        type: 'activateInventorySelection',
        activation,
        click: null,
        shouldClearCanvasInventoryClick: true,
        shouldActivateInventory: true,
        shouldRememberCanvasInventoryClick: false,
        shouldPreventDefault: true,
        shouldReturnEarly: true
      };
    }
    const currentPoint = point || {};
    return {
      handled: true,
      type: 'rememberInventoryActivationClick',
      activation,
      click: {
        key: getInventoryActivationKey(activation),
        x: currentPoint.x,
        y: currentPoint.y,
        time: now
      },
      shouldClearCanvasInventoryClick: false,
      shouldActivateInventory: false,
      shouldRememberCanvasInventoryClick: true,
      shouldPreventDefault: true,
      shouldReturnEarly: true
    };
  }

  function getInventoryActivationSelectionAction(activation) {
    if (!activation) return { handled: false, type: '', id: '' };
    const type = String(activation.type || '');
    const id = String(activation.id || '');
    if (type === 'equipment') return { handled: true, type: 'activateEquipmentItem', id };
    if (type === 'equipped-equipment') return { handled: true, type: 'activateEquippedItem', id };
    if (type === 'consumable') return { handled: true, type: 'activateConsumableItem', id };
    if (type === 'etc' && id === 'cubeFragment') return { handled: true, type: 'openShardCraftPrompt', id };
    return { handled: false, type: '', id };
  }

  function createInventoryActivationUiHelpers() {
    return Object.freeze({
      getCanvasInventoryActivationRegion,
      getInventoryActivationKey,
      isCanvasInventoryDoubleClick,
      getCanvasInventoryActivationClickAction,
      getInventoryActivationSelectionAction
    });
  }

  function getCanvasInventoryDragStartMetadata(payload, activeDrag, point, pointerDown, options) {
    const currentPoint = point || {};
    const startPoint = pointerDown || {};
    const settings = options || {};
    const threshold = Object.prototype.hasOwnProperty.call(settings, 'threshold')
      ? Number(settings.threshold)
      : 5;
    const dragThreshold = Number.isFinite(threshold) ? threshold : 5;
    const startX = Number(startPoint.x || currentPoint.x);
    const startY = Number(startPoint.y || currentPoint.y);
    const distance = Math.hypot(Number(currentPoint.x) - startX, Number(currentPoint.y) - startY);
    const shouldStart = !!payload && !activeDrag && Number.isFinite(distance) && distance > dragThreshold;
    return {
      shouldStart,
      drag: shouldStart ? Object.assign({}, payload, { moved: true, startX, startY }) : null,
      distance,
      threshold: dragThreshold,
      startX,
      startY
    };
  }

  function getCanvasGearDragStartMetadata(uid, activeDrag, point, pointerDown, options) {
    const currentPoint = point || {};
    const startPoint = pointerDown || {};
    const settings = options || {};
    const threshold = Object.prototype.hasOwnProperty.call(settings, 'threshold')
      ? Number(settings.threshold)
      : 5;
    const dragThreshold = Number.isFinite(threshold) ? threshold : 5;
    const gearUid = String(uid || '');
    const startX = Number(startPoint.x || currentPoint.x);
    const startY = Number(startPoint.y || currentPoint.y);
    const distance = Math.hypot(Number(currentPoint.x) - startX, Number(currentPoint.y) - startY);
    const shouldStart = !!gearUid && !activeDrag && Number.isFinite(distance) && distance > dragThreshold;
    return {
      shouldStart,
      drag: shouldStart ? { uid: gearUid, moved: true, startX, startY } : null,
      distance,
      threshold: dragThreshold,
      startX,
      startY
    };
  }

  function getCanvasNoRegionDragSourcePointerAction(region, dragSourceRegion) {
    if (region || !dragSourceRegion) {
      return {
        handled: false,
        type: '',
        region: null,
        shouldBypassModal: false,
        shouldPreventDefault: false
      };
    }
    return {
      handled: true,
      type: 'captureDragSource',
      region: dragSourceRegion,
      shouldBypassModal: true,
      shouldPreventDefault: true
    };
  }

  function getCanvasGearDropAction(target, gearDrag) {
    const dropTarget = target || null;
    const uid = gearDrag && gearDrag.uid;
    const cleanup = {
      shouldClearDrag: true,
      shouldClearCanvasDown: true,
      shouldDraw: true
    };
    if (dropTarget && dropTarget.type === 'upgrade-prompt-drop-zone') {
      return Object.assign({ handled: true, type: 'setUpgradePromptTarget', uid }, cleanup);
    }
    if (dropTarget) return Object.assign({ handled: true, type: 'setPotentialPromptTarget', uid }, cleanup);
    return Object.assign({ handled: true, type: 'showToast', message: 'Drop gear onto the upgrade or prism prompt.' }, cleanup);
  }

  function getCanvasInventoryDropAction(payload) {
    if (!payload) {
      return {
        handled: false,
        type: '',
        payload: null,
        shouldClearDrag: false,
        shouldClearCanvasDown: false,
        shouldDraw: false
      };
    }
    return {
      handled: true,
      type: 'dropInventoryPayloadAtCanvasPoint',
      payload,
      shouldClearDrag: true,
      shouldClearCanvasDown: true,
      shouldDraw: true
    };
  }

  function getCanvasInventoryDragGhostPosition(pointer, bounds) {
    const currentPointer = pointer || {};
    const settings = bounds || {};
    const uiBottom = Number(settings.uiBottom || 0);
    const bottomInset = Object.prototype.hasOwnProperty.call(settings, 'bottomInset')
      ? Number(settings.bottomInset)
      : 50;
    const inset = Number.isFinite(bottomInset) ? bottomInset : 50;
    const x = Number(currentPointer.x || 0) + 10;
    const rawY = Number(currentPointer.y || 0) + 10;
    const maxY = Math.max(10, uiBottom - inset);
    return {
      x,
      y: Math.max(10, Math.min(maxY, rawY))
    };
  }

  function getCanvasInventoryDragGhostMetadata(payload, x, y, options) {
    if (!payload) return null;
    const normalizeInventoryTab = getNormalizeInventoryTab(options);
    const tab = normalizeInventoryTab(payload.tab);
    const source = normalizeInventoryDragSource(payload.source);
    return {
      tab,
      id: String(payload.id || ''),
      source,
      iconKind: tab === 'equipment' ? 'equipment' : tab === 'usable' ? 'usable' : 'material',
      materialSource: source === 'storage' ? 'storage' : 'inventory',
      iconOptions: tab === 'usable' ? { showAura: false } : null,
      alpha: 0.66,
      frame: {
        x: Number(x || 0),
        y: Number(y || 0),
        w: 44,
        h: 44,
        radius: 10,
        fill: 'rgba(238,246,255,0.92)',
        stroke: 'rgba(47,125,214,0.72)'
      },
      icon: {
        x: Number(x || 0) + 7,
        y: Number(y || 0) + 7,
        size: 30
      }
    };
  }

  function createInventoryCanvasDragUiHelpers() {
    return Object.freeze({
      getCanvasInventoryDragStartMetadata,
      getCanvasGearDragStartMetadata,
      getCanvasNoRegionDragSourcePointerAction,
      getCanvasGearDropAction,
      getCanvasInventoryDropAction,
      getCanvasInventoryDragGhostPosition,
      getCanvasInventoryDragGhostMetadata
    });
  }

  function getItemContextMenuActions(context, options) {
    if (!context) return [];
    const settings = options || {};
    const isItemClassBlocked = typeof settings.isItemClassBlocked === 'function'
      ? settings.isItemClassBlocked
      : () => false;
    const canUseItem = typeof settings.canUseItem === 'function'
      ? settings.canUseItem
      : () => true;
    const isStorageContextAvailable = typeof settings.isStorageContextAvailable === 'function'
      ? settings.isStorageContextAvailable
      : () => false;
    const getInventoryTabLabel = typeof settings.getInventoryTabLabel === 'function'
      ? settings.getInventoryTabLabel
      : (tabId) => String(tabId || 'Inventory');
    const isCardDefinitionActive = typeof settings.isCardDefinitionActive === 'function'
      ? settings.isCardDefinitionActive
      : () => false;
    const actions = [];
    const item = context.item || {};
    const source = context.source;
    if (context.kind === 'equipment') {
      const canEquip = !isItemClassBlocked(item) && canUseItem(item);
      if (source === 'storage') {
        actions.push({ id: 'withdraw', label: 'Withdraw' });
      } else {
        actions.push(source === 'equipped'
          ? { id: 'unequip', label: 'Unequip' }
          : { id: 'equip', label: 'Equip', disabled: !canEquip, disabledReason: isItemClassBlocked(item) ? 'Wrong class' : 'Requirement not met' });
        if (isStorageContextAvailable()) actions.push({ id: 'store', label: 'Store' });
        actions.push({ id: 'upgrade', label: 'Upgrade' });
        actions.push({ id: 'attune', label: 'Attune' });
        actions.push({ id: 'toggleLock', label: item.locked ? 'Unlock' : 'Lock' });
        actions.push({ id: 'drop', label: 'Drop', disabled: !!item.locked, disabledReason: 'Unlock before dropping' });
      }
    } else if (context.kind === 'consumable') {
      if (source === 'storage') {
        actions.push({ id: 'withdraw', label: 'Withdraw' });
      } else {
        const useLabel = item.inventorySectionCoupon && item.inventorySectionTab
          ? `Use for ${getInventoryTabLabel(item.inventorySectionTab)}`
          : 'Use';
        actions.push({ id: 'use', label: useLabel });
        actions.push({ id: 'bind', label: 'Bind to Key' });
        if (isStorageContextAvailable()) actions.push({ id: 'store', label: 'Store' });
        actions.push({ id: 'drop', label: 'Drop' });
      }
    } else if (context.kind === 'material') {
      if (source === 'storage') {
        actions.push({ id: 'withdraw', label: 'Withdraw' });
      } else {
        if (context.id === 'cubeFragment') actions.push({ id: 'craftShards', label: 'Combine Shards' });
        if (isStorageContextAvailable()) actions.push({ id: 'store', label: 'Store' });
        actions.push({ id: 'drop', label: 'Drop' });
      }
    } else if (context.kind === 'card') {
      const upgrade = item.combine || item.upgrade || {};
      const sell = item.sell || {};
      const activeDuplicate = isCardDefinitionActive(item);
      actions.push({
        id: 'equipCard',
        label: item.equipped ? 'In Deck' : activeDuplicate ? 'Same Card' : 'Add to Deck',
        disabled: !!activeDuplicate,
        disabledReason: 'This deck already has that card type'
      });
      actions.push({
        id: 'combineCard',
        label: 'Upgrade',
        disabled: !(upgrade.canCombine || upgrade.canUpgrade),
        disabledReason: upgrade.reason || 'Need more copies'
      });
      actions.push({
        id: 'sellCard',
        label: 'Sell',
        disabled: !sell.canSell,
        disabledReason: sell.reason || 'Cannot sell this card'
      });
      actions.push({ id: 'toggleCardLock', label: item.locked ? 'Unlock' : 'Lock' });
    }
    return actions;
  }

  function getOpenItemContextMenuAction(context, x, y, mode, options) {
    const settings = options || {};
    const getActions = typeof settings.getItemContextMenuActions === 'function'
      ? settings.getItemContextMenuActions
      : getItemContextMenuActions;
    if (!context) {
      return {
        handled: false,
        type: '',
        menu: null,
        shouldRenderCommandPanel: false,
        shouldSyncDom: false,
        shouldDraw: false
      };
    }
    const actions = getActions(context);
    if (!actions.length) {
      return {
        handled: false,
        type: '',
        menu: null,
        actions,
        shouldRenderCommandPanel: false,
        shouldSyncDom: false,
        shouldDraw: false
      };
    }
    return {
      handled: true,
      type: 'openItemContextMenu',
      menu: Object.assign({}, context, {
        active: true,
        mode: mode === 'canvas' ? 'canvas' : 'dom',
        x: Number(x || 0),
        y: Number(y || 0),
        actions
      }),
      shouldClearPendingInventoryDrop: true,
      shouldClearDropQuantityPrompt: true,
      shouldClearCanvasInventoryClick: true,
      shouldCloseCommand: true,
      shouldRenderCommandPanel: true,
      shouldSyncDom: true,
      shouldDraw: true
    };
  }

  function getCloseItemContextMenuAction(menu, options) {
    const settings = options || {};
    const hadMenu = !!menu;
    return {
      handled: true,
      type: 'closeContextMenu',
      hadMenu,
      shouldClearMenu: true,
      shouldSyncDom: true,
      shouldDraw: hadMenu && !settings.skipCanvas
    };
  }

  function getItemContextMenuKeyboardAction(event, isDown, menu) {
    const code = event && event.code || '';
    if (!isDown || !menu || code !== 'Escape') {
      return {
        handled: false,
        type: '',
        shouldPreventDefault: false,
        shouldCloseMenu: false
      };
    }
    return {
      handled: true,
      type: 'closeContextMenu',
      shouldPreventDefault: true,
      shouldCloseMenu: true
    };
  }

  function getItemContextMenuDomOpenAction(event, options) {
    const settings = options || {};
    const target = event && event.target;
    const x = event && event.clientX;
    const y = event && event.clientY;
    const empty = {
      handled: false,
      type: '',
      payload: null,
      context: null,
      x,
      y,
      mode: 'dom',
      shouldPreventDefault: false,
      shouldStopPropagation: false,
      shouldCloseMenu: false
    };
    if (!target) return empty;
    const getPayload = typeof settings.getInventoryContextPayloadFromDomTarget === 'function'
      ? settings.getInventoryContextPayloadFromDomTarget
      : () => null;
    const getContext = typeof settings.getItemContextMenuContext === 'function'
      ? settings.getItemContextMenuContext
      : () => null;
    const payload = getPayload(target);
    if (!payload) {
      return Object.assign({}, empty, {
        handled: true,
        type: 'closeContextMenu',
        shouldCloseMenu: true
      });
    }
    const context = getContext(payload);
    if (!context) return Object.assign({}, empty, { payload });
    return {
      handled: true,
      type: 'openContextMenu',
      payload,
      context,
      x,
      y,
      mode: 'dom',
      shouldPreventDefault: true,
      shouldStopPropagation: true,
      shouldCloseMenu: false
    };
  }

  function getItemContextMenuCanvasOpenAction(event, options) {
    const settings = options || {};
    const empty = {
      handled: false,
      type: '',
      point: null,
      payload: null,
      context: null,
      x: null,
      y: null,
      mode: 'canvas',
      shouldPreventDefault: false,
      shouldStopPropagation: false
    };
    if (!event) return empty;
    const getPoint = typeof settings.getCanvasPoint === 'function'
      ? settings.getCanvasPoint
      : () => null;
    const getPayload = typeof settings.getInventoryContextPayloadFromCanvasPoint === 'function'
      ? settings.getInventoryContextPayloadFromCanvasPoint
      : () => null;
    const getContext = typeof settings.getItemContextMenuContext === 'function'
      ? settings.getItemContextMenuContext
      : () => null;
    const point = getPoint(event);
    const payload = getPayload(point);
    const context = getContext(payload);
    if (!context) return Object.assign({}, empty, { point, payload });
    return {
      handled: true,
      type: 'openContextMenu',
      point,
      payload,
      context,
      x: point && point.x,
      y: point && point.y,
      mode: 'canvas',
      shouldPreventDefault: true,
      shouldStopPropagation: true
    };
  }

  function getItemContextMenuExecuteAction(menu, actionId, options) {
    const source = menu || null;
    const actions = source && Array.isArray(source.actions) ? source.actions : [];
    const action = actions.find((candidate) => candidate && candidate.id === actionId);
    if (!source || !action || action.disabled) {
      return {
        handled: false,
        type: '',
        actionId: String(actionId || ''),
        action: action || null,
        payload: null,
        shouldCloseMenu: false
      };
    }
    const settings = options || {};
    const id = source.id;
    const payload = { tab: source.tab, id, index: 0, source: source.source };
    const itemBindPrefix = String(settings.itemBindPrefix || 'item:');
    const potentialPromptConsumableId = settings.potentialPromptConsumableId || 'potential_cube';
    const common = {
      handled: true,
      actionId,
      action,
      id,
      payload,
      shouldCloseMenu: true
    };
    if (actionId === 'equip') return Object.assign({ type: 'activateEquipmentItem' }, common);
    if (actionId === 'unequip') return Object.assign({ type: 'activateEquippedItem' }, common);
    if (actionId === 'use') return Object.assign({ type: 'activateConsumableItem' }, common);
    if (actionId === 'bind') {
      return Object.assign({
        type: 'bindItemToKey',
        bindActionId: `${itemBindPrefix}${id}`,
        keyCode: '',
        toastMessage: 'Choose a key for this item.',
        refreshOptions: { domains: ['session', 'settings'], panel: true, draw: true }
      }, common);
    }
    if (actionId === 'upgrade') return Object.assign({ type: 'openUpgradePrompt' }, common);
    if (actionId === 'attune') {
      return Object.assign({
        type: 'openPotentialPromptAndSetTarget',
        promptUid: '',
        promptOptions: { consumableId: potentialPromptConsumableId, allowEmpty: true }
      }, common);
    }
    if (actionId === 'toggleLock') return Object.assign({ type: 'toggleItemLock', shouldRefreshOnSuccess: true }, common);
    if (actionId === 'equipCard') return Object.assign({ type: 'equipCard' }, common);
    if (actionId === 'combineCard' || actionId === 'upgradeCard') return Object.assign({ type: 'openCardCombinePrompt' }, common);
    if (actionId === 'sellCard') return Object.assign({ type: 'openCardSellPrompt' }, common);
    if (actionId === 'toggleCardLock') return Object.assign({ type: 'toggleCardLock' }, common);
    if (actionId === 'craftShards') return Object.assign({ type: 'openShardCraftPrompt' }, common);
    if (actionId === 'store') return Object.assign({ type: 'storeItemContextPayload' }, common);
    if (actionId === 'withdraw') return Object.assign({ type: 'withdrawItemContextPayload' }, common);
    if (actionId === 'drop') return Object.assign({ type: 'dropItemContextPayload' }, common);
    return Object.assign({ type: 'unknownAction' }, common);
  }

  function getItemContextMenuBox(menu, width, height, options) {
    const clamp = options && typeof options.clamp === 'function'
      ? options.clamp
      : (value, min, high) => Math.max(min, Math.min(high, value));
    const actions = menu && Array.isArray(menu.actions) ? menu.actions : [];
    const w = 190;
    const h = 50 + Math.max(1, actions.length) * 30 + 8;
    return {
      x: clamp(Number(menu && menu.x || 0), 8, Math.max(8, Number(width || 0) - w - 8)),
      y: clamp(Number(menu && menu.y || 0), 8, Math.max(8, Number(height || 0) - h - 8)),
      w,
      h
    };
  }

  function pointInBox(point, rect) {
    return point && rect && point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h;
  }

  function getItemContextMenuCanvasPointerAction(menu, point, width, height, options) {
    const source = menu || null;
    if (!source || source.mode !== 'canvas') {
      return {
        handled: false,
        type: '',
        box: null,
        shouldClearCanvasDown: false,
        shouldPreventDefault: false
      };
    }
    const box = getItemContextMenuBox(source, width, height, options);
    if (pointInBox(point, box)) {
      return {
        handled: false,
        type: '',
        box,
        shouldClearCanvasDown: false,
        shouldPreventDefault: false
      };
    }
    return {
      handled: true,
      type: 'closeContextMenu',
      box,
      shouldClearCanvasDown: true,
      shouldPreventDefault: true
    };
  }

  function getItemContextMenuRegionAction(region) {
    const source = region || {};
    if (source.type === 'item-context-action') return { handled: true, type: 'executeAction', actionId: source.actionId };
    if (source.type === 'item-context-shell') return { handled: true, type: 'ignoreShell' };
    return { handled: false, type: '' };
  }

  function getItemContextMenuActionButton(target, selector) {
    const source = target || null;
    const actionSelector = selector || DOM_ITEM_MENU_ACTION_TARGET_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(actionSelector)
      : null;
  }

  function getItemContextMenuRootTarget(target, selector) {
    const source = target || null;
    const rootSelector = selector || DOM_ITEM_CONTEXT_MENU_ROOT_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(rootSelector)
      : null;
  }

  function getItemContextMenuRootElement(root, selector) {
    const source = root || null;
    const rootSelector = selector || DOM_ITEM_CONTEXT_MENU_ROOT_SELECTOR;
    return source && typeof source.querySelector === 'function'
      ? source.querySelector(rootSelector)
      : null;
  }

  function getItemContextMenuDomClickAction(menu, target, root) {
    if (!menu || menu.mode !== 'dom') {
      return {
        handled: false,
        type: '',
        actionId: '',
        shouldPreventDefault: false,
        shouldCloseMenu: false
      };
    }
    const actionButton = getItemContextMenuActionButton(target);
    if (actionButton && root && typeof root.contains === 'function' && root.contains(actionButton)) {
      return {
        handled: true,
        type: 'executeAction',
        actionId: actionButton.getAttribute ? actionButton.getAttribute(DOM_ITEM_MENU_ACTION_TARGET_ATTRIBUTES[0]) : '',
        shouldPreventDefault: true,
        shouldCloseMenu: false
      };
    }
    const menuRoot = getItemContextMenuRootTarget(target);
    if (menuRoot && root && typeof root.contains === 'function' && root.contains(menuRoot)) {
      return {
        handled: true,
        type: 'ignoreMenu',
        actionId: '',
        shouldPreventDefault: false,
        shouldCloseMenu: false
      };
    }
    return {
      handled: true,
      type: 'closeContextMenu',
      actionId: '',
      shouldPreventDefault: false,
      shouldCloseMenu: true
    };
  }

  function escapeItemContextMenuMarkup(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderItemContextMenuMarkup(menu, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : escapeItemContextMenuMarkup;
    const getBox = typeof settings.getItemContextMenuBox === 'function'
      ? settings.getItemContextMenuBox
      : getItemContextMenuBox;
    const viewportW = Number(settings.viewportWidth || 0);
    const viewportH = Number(settings.viewportHeight || 0);
    const width = viewportW || 1024;
    const height = viewportH || 768;
    const actions = (menu && menu.actions || []).filter(Boolean);
    const box = getBox(menu, width, height);
    return `
        <div class="project-starfall-item-context-menu" data-starfall-item-context-menu-root role="menu" aria-label="${escapeHtml(menu.title || 'Item actions')}" style="left:${box.x}px; top:${box.y}px;">
          <div class="project-starfall-item-context-menu__head">
            <strong>${escapeHtml(menu.title || 'Item')}</strong>
            <span>${escapeHtml(menu.subtitle || '')}</span>
          </div>
          <div class="project-starfall-item-context-menu__actions">
            ${actions.map((action) => `
              <button type="button" data-starfall-item-menu-action="${escapeHtml(action.id)}" role="menuitem" ${action.disabled ? 'disabled' : ''} title="${escapeHtml(action.disabledReason || action.label || '')}">
                ${escapeHtml(action.label || action.id)}
              </button>
            `).join('')}
          </div>
        </div>
      `;
  }

  function getItemContextMenuDomSyncAction(menu, options) {
    const settings = options || {};
    const hasExisting = !!settings.hasExisting;
    const canRemoveExisting = !!settings.canRemoveExisting;
    const canInsertMarkup = !!settings.canInsertMarkup;
    if (!menu || menu.mode !== 'dom') {
      return {
        handled: hasExisting && canRemoveExisting,
        type: hasExisting && canRemoveExisting ? 'removeExisting' : '',
        shouldRemoveExisting: hasExisting && canRemoveExisting,
        shouldRenderMarkup: false,
        shouldReplaceExisting: false,
        shouldInsertMarkup: false
      };
    }
    return {
      handled: true,
      type: hasExisting ? 'replaceExisting' : canInsertMarkup ? 'insertMarkup' : 'renderOnly',
      shouldRemoveExisting: false,
      shouldRenderMarkup: true,
      shouldReplaceExisting: hasExisting,
      shouldInsertMarkup: !hasExisting && canInsertMarkup
    };
  }

  function createInventoryContextMenuUiHelpers() {
    return Object.freeze({
      getItemContextMenuActions,
      getOpenItemContextMenuAction,
      getCloseItemContextMenuAction,
      getItemContextMenuKeyboardAction,
      getItemContextMenuDomOpenAction,
      getItemContextMenuCanvasOpenAction,
      getItemContextMenuExecuteAction,
      getItemContextMenuBox,
      getItemContextMenuCanvasPointerAction,
      getItemContextMenuRegionAction,
      getItemContextMenuActionButton,
      getItemContextMenuRootTarget,
      getItemContextMenuRootElement,
      getItemContextMenuDomClickAction,
      renderItemContextMenuMarkup,
      getItemContextMenuDomSyncAction
    });
  }

  function normalizeStorageCount(value) {
    return Math.max(0, Math.floor(Number(value) || 0));
  }

  function getStorageTransferLabel(payload, options) {
    if (!payload) return 'Item';
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const getEquipmentItem = typeof settings.getEquipmentItem === 'function'
      ? settings.getEquipmentItem
      : () => null;
    const getConsumableItem = typeof settings.getConsumableItem === 'function'
      ? settings.getConsumableItem
      : () => null;
    const getMaterialItem = typeof settings.getMaterialItem === 'function'
      ? settings.getMaterialItem
      : () => null;
    const tab = normalizeInventoryTab(payload.tab);
    if (tab === 'equipment') {
      const source = payload.source === 'storage' ? 'storage' : 'inventory';
      const item = getEquipmentItem(payload.id, source);
      return item && item.name || 'Equipment';
    }
    if (tab === 'usable') {
      const item = getConsumableItem(payload.id);
      return item && item.name || 'Usable item';
    }
    const source = payload.source === 'storage' ? 'storage' : 'inventory';
    const item = getMaterialItem(payload.id, source);
    return item && item.name || 'Etc item';
  }

  function getStorageTransferMax(payload, options) {
    if (!payload) return 0;
    const settings = options || {};
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const normalizeId = typeof settings.normalizeId === 'function'
      ? settings.normalizeId
      : (value) => String(value || '').trim();
    const getSharedStorageStackCount = typeof settings.getSharedStorageStackCount === 'function'
      ? settings.getSharedStorageStackCount
      : () => 0;
    const getInventoryStackCount = typeof settings.getInventoryStackCount === 'function'
      ? settings.getInventoryStackCount
      : () => 0;
    const tab = normalizeInventoryTab(payload.tab);
    if (tab === 'equipment') return 1;
    const id = normalizeId(payload.id);
    return payload.source === 'storage'
      ? normalizeStorageCount(getSharedStorageStackCount(tab, id))
      : normalizeStorageCount(getInventoryStackCount(tab, id));
  }

  function createStorageTransferPrompt(payload, target, targetIndex, options) {
    const settings = options || {};
    const max = Math.max(0, Math.floor(Number(settings.max || 0) || 0));
    if (!payload || max <= 1) return null;
    const normalizeInventoryTab = getNormalizeInventoryTab(settings);
    const normalizeId = typeof settings.normalizeId === 'function'
      ? settings.normalizeId
      : (value) => String(value || '').trim();
    const normalizeSource = typeof settings.normalizeInventoryDragSource === 'function'
      ? settings.normalizeInventoryDragSource
      : normalizeInventoryDragSource;
    const label = settings.label || getStorageTransferLabel(payload, settings);
    const isWithdraw = target === 'inventory';
    return {
      candidate: {
        type: 'storage-transfer',
        id: normalizeId(payload.id),
        tab: normalizeInventoryTab(payload.tab),
        source: normalizeSource(payload.source),
        target: isWithdraw ? 'inventory' : 'storage',
        targetIndex: Math.max(0, Math.floor(Number(targetIndex || 0) || 0)),
        name: label,
        max,
        promptTitle: isWithdraw ? 'Withdraw Stack' : 'Store Stack',
        promptQuestion: isWithdraw ? 'How many should be withdrawn?' : 'How many should be stored?',
        promptConfirm: isWithdraw ? 'Withdraw' : 'Store'
      },
      max,
      quantity: max,
      worldPoint: null
    };
  }

  function createInventoryStorageTransferUiHelpers() {
    return Object.freeze({
      getStorageTransferLabel,
      getStorageTransferMax,
      createStorageTransferPrompt,
      getStorageSlotPayloadDropAction
    });
  }

  const api = {
    DOM_DROP_TARGET_ATTRIBUTES,
    DOM_DROP_TARGET_SELECTOR,
    DOM_INVENTORY_DRAG_TARGET_ATTRIBUTES,
    DOM_INVENTORY_DRAG_TARGET_SELECTOR,
    DOM_GEAR_DRAG_TARGET_ATTRIBUTES,
    DOM_GEAR_DRAG_TARGET_SELECTOR,
    DOM_ITEM_MENU_ACTION_TARGET_ATTRIBUTES,
    DOM_ITEM_MENU_ACTION_TARGET_SELECTOR,
    DOM_ITEM_CONTEXT_MENU_ROOT_ATTRIBUTES,
    DOM_ITEM_CONTEXT_MENU_ROOT_SELECTOR,
    createInventoryDomSelectorUiHelpers,
    normalizeInventoryDragSource,
    getInventoryDragPayloadFromElement,
    getInventoryContextPayloadFromDomTarget,
    getCanvasInventoryDragPayload,
    getInventoryContextPayloadFromCanvasPoint,
    getMaterialContextItem,
    isStorageContextAvailable,
    getItemContextMenuContext,
    createInventoryPayloadContextUiHelpers,
    getStoreItemContextPayloadAction,
    getWithdrawItemContextPayloadAction,
    getDropItemContextPayloadAction,
    createInventoryContextPayloadActionUiHelpers,
    getStorageSlotPayloadDropAction,
    getInventorySlotPayloadDropAction,
    getInventoryPayloadCanvasDropAction,
    getCanvasGearDragUid,
    createInventoryPayloadDropUiHelpers,
    serializeInventoryDragPayload,
    parseInventoryDragPayload,
    getInventoryDragPayloadEventReadAction,
    getInventoryDragPayloadFromEventData,
    getNativeDragPreviewCleanupAction,
    getNativeDragPreviewSetupAction,
    getNativeDragLabel,
    getNativeDragPreviewMetadata,
    getNativeDragIconCloneMetadata,
    createInventoryNativeDragUiHelpers,
    getDomInventoryDragStartAction,
    getDomInventoryDragStartGateAction,
    getDomGearDragStartAction,
    getDomGearDragStartGateAction,
    getDomInventoryDragStartStateAction,
    getDomInventoryDragStartTransferWriteAction,
    getDomInventoryDragEndCleanupAction,
    getDomDropTargetAction,
    getDomDragOverAction,
    getDomCanvasOutsideDropAction,
    getDomTargetDropStartAction,
    getDomSlotPayloadDropAction,
    getDomPromptDropTransferReadAction,
    getDomPromptDropAction,
    createInventoryDomDragDropUiHelpers,
    getDropCandidateForEquipment,
    getDropCandidateForConsumable,
    getDropCandidateForMaterial,
    getDropCandidateForCanvasRegion,
    getDropCandidateForInventoryPayload,
    getDropQuantityMax,
    normalizeDropPromptQuantity,
    createDropQuantityPrompt,
    isPendingInventoryDrop,
    createInventoryDropCandidateUiHelpers,
    getCanvasInventoryActivationRegion,
    getInventoryActivationKey,
    isCanvasInventoryDoubleClick,
    getCanvasInventoryActivationClickAction,
    getInventoryActivationSelectionAction,
    createInventoryActivationUiHelpers,
    getCanvasInventoryDragStartMetadata,
    getCanvasGearDragStartMetadata,
    getCanvasNoRegionDragSourcePointerAction,
    getCanvasGearDropAction,
    getCanvasInventoryDropAction,
    getCanvasInventoryDragGhostPosition,
    getCanvasInventoryDragGhostMetadata,
    createInventoryCanvasDragUiHelpers,
    getItemContextMenuActions,
    getOpenItemContextMenuAction,
    getCloseItemContextMenuAction,
    getItemContextMenuKeyboardAction,
    getItemContextMenuDomOpenAction,
    getItemContextMenuCanvasOpenAction,
    getItemContextMenuExecuteAction,
    getItemContextMenuBox,
    getItemContextMenuCanvasPointerAction,
    getItemContextMenuRegionAction,
    getItemContextMenuActionButton,
    getItemContextMenuRootTarget,
    getItemContextMenuRootElement,
    getItemContextMenuDomClickAction,
    renderItemContextMenuMarkup,
    getItemContextMenuDomSyncAction,
    createInventoryContextMenuUiHelpers,
    getStorageTransferLabel,
    getStorageTransferMax,
    createStorageTransferPrompt,
    createInventoryStorageTransferUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.inventoryDragDrop = Object.assign({}, modules.inventoryDragDrop || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
