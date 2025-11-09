(() => {
  'use strict';
  const modal = document.getElementById('contact-modal');
  const content = modal?.querySelector('.modal-content');
  const openBtn = document.getElementById('contact-form-toggle');
  const closeBtn = modal?.querySelector('.modal-close');
  let prevFocus = null;

  const focusables = () => content.querySelectorAll('a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])');
  const trap = (e) => {
    if (e.key !== 'Tab') return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0], last = f[f.length-1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  function open(){ if(!modal) return;
    prevFocus = document.activeElement;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    content.setAttribute('tabindex','0');
    content.focus({preventScroll:true});
    content.addEventListener('keydown', trap);
  }
  function close(){ if(!modal) return;
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    content.removeEventListener('keydown', trap);
    if (prevFocus) prevFocus.focus();
  }
  const openIfHashMatches = () => {
    if (!modal) return;
    if (location.hash === '#contact-modal') {
      // Slight delay so layout settles before opening
      setTimeout(() => {
        if (!modal.classList.contains('active')) open();
      }, 120);
    }
  };

  openBtn && openBtn.addEventListener('click', open);
  closeBtn && closeBtn.addEventListener('click', close);
  modal && modal.addEventListener('click', (e)=>{ if(e.target === modal) close(); });
  document.addEventListener('keydown', (e)=>{ if(e.key === 'Escape' && modal.classList.contains('active')) close(); });
  document.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-contact-modal-link]');
    if (!trigger) return;
    event.preventDefault();
    open();
  });
  window.addEventListener('hashchange', openIfHashMatches);
  openIfHashMatches();
})();
