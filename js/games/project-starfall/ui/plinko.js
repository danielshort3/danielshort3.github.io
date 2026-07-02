(function initProjectStarfallUiPlinko(global) {
  'use strict';

  const DOM_PLINKO_DROP_TARGET_ATTRIBUTES = Object.freeze([
    'data-starfall-plinko-drop'
  ]);
  const DOM_PLINKO_DROP_TARGET_SELECTOR = DOM_PLINKO_DROP_TARGET_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');
  const PLINKO_HOLD_INITIAL_DELAY_MS = 300;
  const PLINKO_HOLD_REPEAT_MS = 180;

  function getPlinkoPanelDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const selectId = getAttribute('data-starfall-plinko-select');
    if (selectId) return { handled: true, type: 'selectBall', ballId: selectId };
    const buyQuantityId = getAttribute('data-starfall-plinko-buy-quantity');
    if (buyQuantityId) {
      return {
        handled: true,
        type: 'adjustBuyQuantity',
        ballId: buyQuantityId,
        delta: Math.floor(Number(getAttribute('data-starfall-plinko-buy-quantity-delta') || 0) || 0)
      };
    }
    const buyId = getAttribute('data-starfall-plinko-buy');
    if (buyId) {
      return {
        handled: true,
        type: 'buyBall',
        ballId: buyId,
        amountValue: getAttribute('data-starfall-plinko-buy-amount')
      };
    }
    const dropId = getAttribute(DOM_PLINKO_DROP_TARGET_ATTRIBUTES[0]);
    if (dropId) return { handled: true, type: 'dropBall', ballId: dropId };
    if (hasAttribute('data-starfall-plinko-claim-all')) return { handled: true, type: 'claimAll' };
    return { handled: false, type: '' };
  }

  function getPlinkoDropTarget(target, selector) {
    const source = target && typeof target.closest === 'function'
      ? target.closest(selector || DOM_PLINKO_DROP_TARGET_SELECTOR)
      : null;
    return source || null;
  }

  function getPlinkoDropPointerDomAction(target) {
    const source = getPlinkoDropTarget(target);
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const ballId = getAttribute(DOM_PLINKO_DROP_TARGET_ATTRIBUTES[0]);
    if (!ballId) return { handled: false, type: '', target: source };
    return {
      handled: true,
      type: 'startDropHold',
      ballId,
      target: source,
      disabled: !!(source && source.disabled)
    };
  }

  function getPlinkoDomDropHoldStartAction(pointerAction, options) {
    const source = pointerAction || {};
    const settings = options || {};
    const handled = !!(source.handled && source.ballId && settings.inRoot && !source.disabled);
    return {
      handled,
      type: handled ? 'startDropHold' : '',
      ballId: source.ballId || '',
      target: source.target || null,
      disabled: !!source.disabled,
      inRoot: !!settings.inRoot,
      shouldStartDropHold: handled,
      startOptions: { source: 'dom' },
      shouldPreventDefaultOnStart: handled
    };
  }

  function getPlinkoPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'plinko-select') return { handled: true, type: 'selectBall', ballId: source.ballId };
    if (source.type === 'plinko-buy-quantity') {
      return { handled: true, type: 'adjustBuyQuantity', ballId: source.ballId, delta: source.delta };
    }
    if (source.type === 'plinko-buy') {
      return { handled: true, type: 'buyBall', ballId: source.ballId, quantity: source.quantity };
    }
    if (source.type === 'plinko-drop') return { handled: true, type: 'dropBall', ballId: source.ballId };
    if (source.type === 'plinko-claim-all') return { handled: true, type: 'claimAll' };
    return { handled: false, type: '' };
  }

  function getPlinkoCanvasPointerAction(region) {
    const source = region || {};
    if (source.type !== 'plinko-drop') {
      return { handled: false, type: '', ballId: '', region: null, shouldPreventDefault: false };
    }
    return {
      handled: true,
      type: 'startDropHold',
      ballId: source.ballId,
      region: source,
      shouldPreventDefault: true
    };
  }

  function getPlinkoCanvasReleaseAction(region) {
    const source = region || {};
    if (source.type !== 'plinko-drop') {
      return {
        handled: false,
        type: '',
        ballId: '',
        shouldConsumeHoldClick: false,
        shouldStopHold: false,
        stopOptions: null
      };
    }
    return {
      handled: true,
      type: 'releaseDropHold',
      ballId: source.ballId,
      shouldConsumeHoldClick: true,
      shouldStopHold: true,
      stopOptions: { skipDraw: true }
    };
  }

  function createPlinkoInteractionUiHelpers() {
    return Object.freeze({
      getPlinkoPanelDomAction,
      getPlinkoDropTarget,
      getPlinkoDropPointerDomAction,
      getPlinkoDomDropHoldStartAction,
      getPlinkoPanelRegionAction,
      getPlinkoCanvasPointerAction,
      getPlinkoCanvasReleaseAction
    });
  }

  function createPlinkoDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_PLINKO_DROP_TARGET_ATTRIBUTES,
      DOM_PLINKO_DROP_TARGET_SELECTOR
    });
  }

  const api = {
    DOM_PLINKO_DROP_TARGET_ATTRIBUTES,
    DOM_PLINKO_DROP_TARGET_SELECTOR,
    PLINKO_HOLD_INITIAL_DELAY_MS,
    PLINKO_HOLD_REPEAT_MS,
    createPlinkoInteractionUiHelpers,
    createPlinkoDomSelectorUiHelpers,
    getPlinkoPanelDomAction,
    getPlinkoDropTarget,
    getPlinkoDropPointerDomAction,
    getPlinkoDomDropHoldStartAction,
    getPlinkoPanelRegionAction,
    getPlinkoCanvasPointerAction,
    getPlinkoCanvasReleaseAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.plinko = Object.assign({}, modules.plinko || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
