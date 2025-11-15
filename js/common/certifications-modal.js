(() => {
  'use strict';
  const modal = document.getElementById('certifications-modal');
  const modalContent = modal?.querySelector('.modal-content');
  const openers = document.querySelectorAll('[data-cert-modal-open]');
  const closers = modal ? modal.querySelectorAll('[data-cert-modal-close], .modal-close') : [];
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

  const openModal = () => {
    previousFocus = document.activeElement;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    modalContent.focus({ preventScroll: true });
    modalContent.addEventListener('keydown', trapFocus);
  };

  const closeModal = () => {
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    modalContent.removeEventListener('keydown', trapFocus);
    if (previousFocus && typeof previousFocus.focus === 'function') {
      previousFocus.focus();
    }
  };

  openers.forEach((btn) => btn.addEventListener('click', (event) => {
    event.preventDefault();
    openModal();
  }));
  closers.forEach((btn) => btn.addEventListener('click', closeModal));
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeModal();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
})();
