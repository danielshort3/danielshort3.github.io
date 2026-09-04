(() => {
  'use strict';

  const MODAL_PARAM = 'modal';
  const MODAL_VALUE = 'certifications';
  const OPEN_SELECTOR = '[data-cert-modal-open]';
  const CLOSE_SELECTOR = '[data-cert-modal-close], .modal-close';
  let activeController = null;

  const findWithin = (root, selector) => {
    if (!root) return null;
    if (typeof root.matches === 'function' && root.matches(selector)) return root;
    return typeof root.querySelector === 'function' ? root.querySelector(selector) : null;
  };

  const currentElements = (root = document) => {
    const modal = findWithin(root, '#certifications-modal') || document.getElementById('certifications-modal');
    const modalContent = modal?.querySelector('.modal-content');
    let openers = Array.from(document.querySelectorAll(OPEN_SELECTOR));
    if (!openers.length && root !== document && typeof root?.querySelectorAll === 'function') {
      openers = Array.from(root.querySelectorAll(OPEN_SELECTOR));
    }
    const closers = modal ? Array.from(modal.querySelectorAll(CLOSE_SELECTOR)) : [];
    return { modal, modalContent, openers, closers };
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

  const createController = ({ modal, modalContent, openers, closers }) => {
    const modalAccessibility = typeof window.createModalAccessibility === 'function'
      ? window.createModalAccessibility(modal)
      : null;
    if (!modalAccessibility) return null;

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
      previousFocus = document.activeElement;
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
      modalAccessibility.restoreBackground();
      modal.classList.remove('active');
      modalAccessibility.hide();
      if (!document.querySelector('.modal.active')) {
        document.body.classList.remove('modal-open');
      }
      modalContent.removeEventListener('keydown', trapFocus);
      if (options.restoreFocus !== false) {
        const focusTarget = previousFocus
          && previousFocus !== document.body
          && document.contains(previousFocus)
          && typeof previousFocus.focus === 'function'
          ? previousFocus
          : openers.find((opener) => document.contains(opener));
        if (focusTarget && typeof focusTarget.focus === 'function') {
          focusTarget.focus({ preventScroll: true });
        }
      }
      previousFocus = null;
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

    return {
      modal,
      matches(elements) {
        return modal === elements.modal &&
          modalContent === elements.modalContent &&
          sameElements(openers, elements.openers) &&
          sameElements(closers, elements.closers);
      },
      dispose() {
        if (disposed) return;
        closeModal({ skipURL: true, restoreFocus: false });
        disposed = true;
        modalContent.removeEventListener('keydown', trapFocus);
        cleanup.splice(0).forEach((remove) => remove());
      }
    };
  };

  const mountCertificationsModal = (root = document) => {
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
