(() => {
  'use strict';

  const MODAL_PARAM = 'modal';
  const MODAL_VALUE = 'certifications';
  const OPEN_SELECTOR = '[data-cert-modal-open]';
  const CLOSE_SELECTOR = '[data-cert-modal-close], .modal-close';
  const ROUTE_CONTENT_SELECTOR = '[data-site-route-content], [data-personal-detail-content]';
  let activeController = null;

  const findWithin = (root, selector) => {
    if (!root) return null;
    if (typeof root.matches === 'function' && root.matches(selector)) return root;
    return typeof root.querySelector === 'function' ? root.querySelector(selector) : null;
  };

  const currentElements = (root = document) => {
    const belongsToRoot = activeController && (root === document ||
      activeController.ownerRoot === root || root.contains?.(activeController.ownerRoot));
    const modal = findWithin(root, '#certifications-modal') || (belongsToRoot ? activeController.modal : null);
    const modalContent = modal?.querySelector('.modal-content');
    let openers = Array.from(document.querySelectorAll(OPEN_SELECTOR));
    if (!openers.length && root !== document && typeof root?.querySelectorAll === 'function') {
      openers = Array.from(root.querySelectorAll(OPEN_SELECTOR));
    }
    const closers = modal ? Array.from(modal.querySelectorAll(CLOSE_SELECTOR)) : [];
    const ownerRoot = modal === activeController?.modal ? activeController.ownerRoot : root === document
      ? modal?.closest(ROUTE_CONTENT_SELECTOR) || document.querySelector(ROUTE_CONTENT_SELECTOR) || document
      : root;
    return { modal, modalContent, openers, closers, ownerRoot };
  };

  const sameElements = (left, right) => left.length === right.length &&
    left.every((node, index) => node === right[index]);

  const urlHasCertModal = () => {
    try {
      return new URLSearchParams(window.location.search).get(MODAL_PARAM) === MODAL_VALUE;
    } catch {
      return false;
    }
  };

  const updateURLState = (shouldOpen) => {
    if (!window.history?.replaceState) return;
    try {
      const url = new URL(window.location.href);
      const params = new URLSearchParams(url.search);
      if (shouldOpen) {
        params.set(MODAL_PARAM, MODAL_VALUE);
      } else {
        params.delete(MODAL_PARAM);
      }
      const qs = params.toString();
      const next = `${url.pathname}${qs ? `?${qs}` : ''}${url.hash || ''}`;
      window.history.replaceState(null, '', next);
    } catch {}
  };

  const createController = ({ modal, modalContent, openers, closers, ownerRoot }) => {
    if (typeof window.createModalAccessibility !== 'function') return null;
    const placeholder = modal.parentElement !== document.body
      ? document.createComment('certifications-modal')
      : null;
    if (placeholder) {
      modal.before(placeholder);
      // Preserve the original dialog while escaping the tab panel's stacking context.
      document.body.appendChild(modal);
    }
    const restorePortal = () => {
      if (placeholder) {
        if (placeholder.isConnected) placeholder.replaceWith(modal);
        else {
          modal.remove();
          placeholder.remove();
        }
      } else if (ownerRoot !== document) {
        modal.remove();
      }
    };
    const modalAccessibility = window.createModalAccessibility(modal);
    if (!modalAccessibility) {
      restorePortal();
      return null;
    }

    const cleanup = [];
    const focusableSelectors = 'a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])';
    let previousFocus = null;
    let disposed = false;

    const listen = (target, type, listener) => {
      target.addEventListener(type, listener);
      cleanup.push(() => target.removeEventListener(type, listener));
    };

    const getFocusables = () => Array.from(modalContent.querySelectorAll(focusableSelectors))
      .filter((node) => !node.disabled && !node.closest('[hidden], [inert], [aria-hidden="true"]'));

    const trapFocus = (event) => {
      if (event.key !== 'Tab') return;
      const nodes = getFocusables();
      if (!nodes.length) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    const openModal = (options = {}) => {
      if (disposed || modal.classList.contains('active')) return;
      if (!modal.contains(document.activeElement)) previousFocus = document.activeElement;
      modalAccessibility.show();
      modal.classList.add('active');
      document.body.classList.add('modal-open');
      modalContent.focus({ preventScroll: true });
      modalAccessibility.isolateBackground();
      modalContent.addEventListener('keydown', trapFocus);
      if (!options.skipURL) updateURLState(true);
    };

    const closeModal = (options = {}) => {
      if (!modal.classList.contains('active')) return;
      modalAccessibility.hide({
        immediate: options.immediate === true,
        onFinish: () => {
          modalContent.removeEventListener('keydown', trapFocus);
          if (options.restoreFocus !== false) {
            const focusTarget = previousFocus
              && previousFocus !== document.body
              && document.contains(previousFocus)
              && typeof previousFocus.focus === 'function'
              ? previousFocus
              : openers.find((opener) => document.contains(opener));
            focusTarget?.focus({ preventScroll: true });
          }
          previousFocus = null;
        }
      });
      if (!options.skipURL) updateURLState(false);
    };

    const syncWithURL = () => {
      const wantsOpen = urlHasCertModal();
      const isOpen = modal.classList.contains('active');
      if (wantsOpen && !isOpen) {
        openModal({ skipURL: true });
      } else if (!wantsOpen && isOpen) {
        closeModal({ skipURL: true });
      }
    };

    const handleOpen = (event) => {
      event.preventDefault();
      openModal();
    };
    const handleClose = () => closeModal();
    const handleBackdrop = (event) => {
      if (event.target === modal) closeModal();
    };
    const handleEscape = (event) => {
      if (event.key === 'Escape' && modal.classList.contains('active')) closeModal();
    };

    openers.forEach((opener) => listen(opener, 'click', handleOpen));
    closers.forEach((closer) => listen(closer, 'click', handleClose));
    listen(modal, 'click', handleBackdrop);
    listen(document, 'keydown', handleEscape);
    listen(window, 'popstate', syncWithURL);
    listen(window, 'hashchange', syncWithURL);
    syncWithURL();

    const controller = {
      modal,
      ownerRoot,
      matches(elements) {
        return modal === elements.modal &&
          modalContent === elements.modalContent &&
          sameElements(openers, elements.openers) &&
          sameElements(closers, elements.closers);
      },
      dispose() {
        if (disposed) return;
        closeModal({ skipURL: true, restoreFocus: false, immediate: true });
        modalAccessibility.dispose?.();
        disposed = true;
        modalContent.removeEventListener('keydown', trapFocus);
        cleanup.splice(0).forEach((remove) => remove());
        restorePortal();
        if (activeController === controller) activeController = null;
      }
    };
    listen(document, 'site:route-unmounted', (event) => {
      if (event.detail?.root === ownerRoot) controller.dispose();
    });
    return controller;
  };

  const mountCertificationsModal = (root = document) => {
    if (activeController?.ownerRoot !== document && activeController?.ownerRoot?.isConnected === false) {
      activeController.dispose();
    }
    const elements = currentElements(root);
    if (!elements.modal || !elements.modalContent || !elements.openers.length) {
      activeController?.dispose();
      activeController = null;
      return null;
    }
    if (activeController?.matches(elements)) return activeController;

    activeController?.dispose();
    activeController = createController(elements);
    return activeController;
  };

  const remountForCurrentRoute = (event) => {
    mountCertificationsModal(event.detail?.root || document);
  };
  document.addEventListener('site:route-mounted', remountForCurrentRoute);
  document.addEventListener('site:content-updated', remountForCurrentRoute);

  const initializeDirect = () => mountCertificationsModal(document);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDirect, { once: true });
  } else {
    initializeDirect();
  }
})();
