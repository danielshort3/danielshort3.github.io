(() => {
  'use strict';
  function openContactModal() {
    const modal = document.getElementById('contact-modal');
    if (!modal) return;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    const focusable = modal.querySelectorAll('a,button,[tabindex]:not([tabindex="-1"])');
    focusable[0]?.focus();

    const trap = e => {
      if (e.key === 'Escape') { close(); return; }
      if (e.key !== 'Tab' || !focusable.length) return;
      const first = focusable[0],
            last  = focusable[focusable.length - 1];
      if (e.shiftKey ? document.activeElement === first
                     : document.activeElement === last) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      }
    };

    const clickClose = e => {
      if (e.target.classList.contains('modal') ||
          e.target.classList.contains('modal-close')) close();
    };

    const close = () => {
      modal.classList.remove('active');
      document.body.classList.remove('modal-open');
      document.removeEventListener('keydown', trap);
      modal.removeEventListener('click', clickClose);
    };

    document.addEventListener('keydown', trap);
    modal.addEventListener('click', clickClose);
  }

  function initContactModal() {
    if (document.body.dataset.page !== 'contact') return;
    const btn = document.getElementById('contact-form-toggle');
    if (btn) btn.addEventListener('click', openContactModal);
  }

  document.addEventListener('DOMContentLoaded', initContactModal);
})();
