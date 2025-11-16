(() => {
  'use strict';
  const modal = document.getElementById('certifications-modal');
  const modalContent = modal?.querySelector('.modal-content');
  const openers = document.querySelectorAll('[data-cert-modal-open]');
  const closers = modal ? modal.querySelectorAll('[data-cert-modal-close], .modal-close') : [];
  const MODAL_PARAM = 'modal';
  const MODAL_VALUE = 'certifications';
  let previousFocus = null;

  if (!modal || !modalContent || !openers.length) return;

  const focusableSelectors = 'a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])';
  const getFocusables = () => modalContent.querySelectorAll(focusableSelectors);
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

  const urlHasCertModal = () => {
    try {
      return new URLSearchParams(window.location.search).get(MODAL_PARAM) === MODAL_VALUE;
    } catch {
      return false;
    }
  };

  const updateURLState = (shouldOpen) => {
    if (!history?.replaceState) return;
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
      history.replaceState(null, '', next);
    } catch {}
  };

  const openModal = (opts = {}) => {
    if (modal.classList.contains('active')) return;
    previousFocus = document.activeElement;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    modalContent.focus({ preventScroll: true });
    modalContent.addEventListener('keydown', trapFocus);
    if (!opts.skipURL) updateURLState(true);
  };

  const closeModal = (opts = {}) => {
    if (!modal.classList.contains('active')) return;
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    modalContent.removeEventListener('keydown', trapFocus);
    if (previousFocus && typeof previousFocus.focus === 'function') {
      previousFocus.focus();
    }
    if (!opts.skipURL) updateURLState(false);
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

  openers.forEach((btn) => btn.addEventListener('click', (event) => {
    event.preventDefault();
    openModal();
  }));
  closers.forEach((btn) => btn.addEventListener('click', () => closeModal()));
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeModal();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  window.addEventListener('popstate', syncWithURL);
  window.addEventListener('hashchange', syncWithURL);
  syncWithURL();
})();
