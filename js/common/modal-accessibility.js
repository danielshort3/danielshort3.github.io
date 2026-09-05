(() => {
  'use strict';

  if (window.createModalAccessibility) return;
  const controllers = new WeakMap();
  const visibleModals = new Set();
  let backgroundOwner = null;

  const restoreAttribute = (element, name, value) => {
    if (value === null) element.removeAttribute(name);
    else element.setAttribute(name, value);
  };

  const createModalAccessibility = (modal) => {
    if (controllers.has(modal)) return controllers.get(modal);
    const backgroundState = new Map();
    const setOpenState = (isOpen) => {
      modal.hidden = !isOpen;
      modal.inert = !isOpen;
      modal.toggleAttribute('inert', !isOpen);
      modal.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
    };
    const restoreBackground = () => {
      backgroundState.forEach((state, element) => {
        element.inert = state.hadInert;
        element.toggleAttribute('inert', state.hadInert);
        restoreAttribute(element, 'aria-hidden', state.ariaHidden);
      });
      backgroundState.clear();
      if (backgroundOwner === controller) backgroundOwner = null;
    };
    const isolateBackground = () => {
      if (backgroundOwner && backgroundOwner !== controller) backgroundOwner.restoreBackground();
      if (backgroundState.size) return;
      backgroundOwner = controller;
      let current = modal;
      while (current && current.parentElement) {
        const parent = current.parentElement;
        Array.from(parent.children).forEach((sibling) => {
          if (sibling === current || backgroundState.has(sibling)) return;
          backgroundState.set(sibling, {
            hadInert: sibling.hasAttribute('inert'),
            ariaHidden: sibling.getAttribute('aria-hidden')
          });
          sibling.inert = true;
          sibling.setAttribute('inert', '');
          sibling.setAttribute('aria-hidden', 'true');
        });
        if (parent === document.body) break;
        current = parent;
      }
    };
    const syncScrollLock = () => {
      visibleModals.forEach((element) => {
        if (!element.isConnected) visibleModals.delete(element);
      });
      document.body?.classList.toggle('modal-open', visibleModals.size > 0 || Boolean(document.querySelector('.modal.active')));
    };
    const controller = {
      show() {
        if (backgroundOwner && backgroundOwner !== controller) backgroundOwner.restoreBackground();
        modal.inert = false;
        modal.removeAttribute('inert');
        modal.setAttribute('aria-hidden', 'false');
        visibleModals.add(modal);
        syncScrollLock();
        if (window.SiteMotion) {
          return window.SiteMotion.presence(modal, true, { className: 'active', enter: '--motion-slow', exit: '--motion-base' });
        }
        setOpenState(true);
        return null;
      },
      hide({ onFinish, immediate = false } = {}) {
        const complete = () => {
          restoreBackground();
          // Move focus before making its former subtree inert/hidden.
          onFinish?.();
          setOpenState(false);
          visibleModals.delete(modal);
          if (!backgroundOwner) {
            const parentModal = [...visibleModals].reverse().find((element) => element.isConnected && element.classList.contains('active'));
            if (parentModal) controllers.get(parentModal)?.isolateBackground();
          }
          syncScrollLock();
        };
        if (window.SiteMotion && !immediate) {
          return window.SiteMotion.presence(modal, false, {
            className: 'active', enter: '--motion-slow', exit: '--motion-base', hidden: false, onFinish: complete
          });
        }
        window.SiteMotion?.cancel(modal);
        modal.classList.remove('active');
        complete();
        return null;
      },
      isolateBackground,
      restoreBackground,
      dispose() {
        controller.hide({ immediate: true });
        controllers.delete(modal);
      }
    };
    setOpenState(modal.classList.contains('active'));
    controllers.set(modal, controller);
    return controller;
  };

  new MutationObserver(() => {
    visibleModals.forEach((modal) => {
      if (!modal.isConnected) controllers.get(modal)?.dispose();
    });
  }).observe(document.documentElement, { childList: true, subtree: true });

  window.createModalAccessibility = createModalAccessibility;
})();
