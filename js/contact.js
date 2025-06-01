(() => {
  'use strict';
  function initContactFormToggle() {
    if (document.body.dataset.page !== 'contact') return;
    const btn = document.getElementById('contact-form-toggle');
    const formWrap = document.getElementById('contact-form');
    if (!btn || !formWrap) return;
    const pad = parseFloat(getComputedStyle(formWrap).paddingTop) || 32;

    btn.addEventListener('click', () => {
      const expanded = btn.dataset.expanded === 'true';
      btn.dataset.expanded = expanded ? 'false' : 'true';
      btn.textContent = expanded ? 'Send a Message' : 'Hide Form';
      if (expanded) {
        const start = formWrap.offsetHeight;
        formWrap.style.height = `${start}px`;
        formWrap.classList.add('grid-fade');
        requestAnimationFrame(() => {
          formWrap.style.height = '0px';
          formWrap.style.paddingTop = '0px';
          formWrap.style.paddingBottom = '0px';
        });
        setTimeout(() => {
          formWrap.classList.add('hide');
          formWrap.classList.remove('grid-fade');
          formWrap.style.height = '';
          formWrap.style.paddingTop = '';
          formWrap.style.paddingBottom = '';
        }, 450);
      } else {
        formWrap.classList.remove('hide');
        formWrap.classList.add('active');
        const target = formWrap.scrollHeight;
        formWrap.style.height = '0px';
        formWrap.style.paddingTop = '0px';
        formWrap.style.paddingBottom = '0px';
        formWrap.classList.add('grid-fade');
        requestAnimationFrame(() => {
          formWrap.style.height = `${target}px`;
          formWrap.style.paddingTop = `${pad}px`;
          formWrap.style.paddingBottom = `${pad}px`;
        });
        setTimeout(() => {
          formWrap.classList.remove('grid-fade');
          formWrap.style.height = '';
          formWrap.style.paddingTop = '';
          formWrap.style.paddingBottom = '';
        }, 450);
      }
      if (window.gaEvent) {
        window.gaEvent('contact_form_toggle', { expanded: !expanded });
      }
    });
  }
  document.addEventListener('DOMContentLoaded', initContactFormToggle);
})();
