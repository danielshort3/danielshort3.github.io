
/* ===================================================================
   File: contact.js
   Purpose: Handles the contact form modal and tracking
   =================================================================== */
( () => {
  'use strict';

  // Adjust iframe height so the form fits nicely in the modal
  function setContactModalHeight() {
    const modal = document.getElementById('contact-modal');
    if (!modal) return;
    const content = modal.querySelector('.modal-content');
    const body    = modal.querySelector('.modal-body');
    const iframe  = modal.querySelector('iframe');
    const header  = modal.querySelector('.modal-title-strip');
    if (!content || !body || !iframe || !header) return;

    const bodyStyles = getComputedStyle(body);
    const padding =
      parseFloat(bodyStyles.paddingTop) + parseFloat(bodyStyles.paddingBottom);
    const chrome = header.getBoundingClientRect().height + padding;

    const max = window.innerHeight * 0.82;

    let iframeHeight = 0;
    try {
      const doc = iframe.contentDocument || iframe.contentWindow.document;
      iframeHeight = doc.documentElement.scrollHeight;
    } catch (err) {
      iframeHeight = 0;
    }

    if (!iframeHeight) iframeHeight = max - chrome;
    iframe.style.height = `${Math.min(iframeHeight, max - chrome)}px`;

    if (content) content.style.maxHeight = `${max}px`;
  }
  // Display the modal and wire up focus trapping
  function openContactModal() {
    const modal = document.getElementById('contact-modal');
    if (!modal) return;
    setContactModalHeight();
    modal.classList.add('active');
    if (window.gaEvent) window.gaEvent('contact_form_open');
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

  // Attach handlers once the contact page is ready
  function initContactModal() {
    if (document.body.dataset.page !== 'contact') return;
    const btn = document.getElementById('contact-form-toggle');
    if (btn) btn.addEventListener('click', openContactModal);
    window.addEventListener('resize', setContactModalHeight);
    setContactModalHeight();
    initFormSubmitTracking();
  }

  // Track successful form submissions inside the iframe
  function initFormSubmitTracking() {
    const iframe = document.querySelector('#contact-modal iframe');
    if (!iframe) return;
    iframe.addEventListener('load', () => {
      try {
        const href = iframe.contentWindow.location.href;
        if (href.includes('formResponse')) {
          if (window.gaEvent) window.gaEvent('contact_form_submit');
        }
      } catch (err) {
        // ignore cross-origin access errors
      }
    });
  }

  // Initialise when the DOM is ready
  document.addEventListener('DOMContentLoaded', initContactModal);
})();
