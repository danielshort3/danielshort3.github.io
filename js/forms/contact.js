(() => {
  'use strict';
  const modal = document.getElementById('contact-modal');
  const content = modal?.querySelector('.modal-content');
  const openBtn = document.getElementById('contact-form-toggle');
  const closeBtn = modal?.querySelector('.modal-close');
  const form = document.getElementById('contact-form');
  const statusEl = document.getElementById('contact-status');
  const submitBtn = form?.querySelector('[type="submit"]');
  const endpoint = form?.dataset.endpoint || form?.getAttribute('action') || '';
  let prevFocus = null;
  let sending = false;

  const focusables = () => content.querySelectorAll('a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])');
  const trap = (e) => {
    if (e.key !== 'Tab') return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0], last = f[f.length-1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  const setStatus = (message = '', tone = 'info') => {
    if (!statusEl) return;
    statusEl.textContent = message;
    if (message) statusEl.dataset.tone = tone;
    else delete statusEl.dataset.tone;
  };

  const resetForm = () => {
    if (form) form.reset();
    sending = false;
    if (submitBtn) submitBtn.disabled = false;
    setStatus('');
  };

  function open(){ if(!modal) return;
    resetForm();
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
  modal?.querySelectorAll('[data-contact-close]')?.forEach(btn => btn.addEventListener('click', close));
  modal && modal.addEventListener('click', (e)=>{ if(e.target === modal) close(); });
  document.addEventListener('keydown', (e)=>{ if(e.key === 'Escape' && modal.classList.contains('active')) close(); });
  document.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-contact-modal-link]');
    if (!trigger) return;
    event.preventDefault();
    open();
  });
  if (form) {
    form.addEventListener('submit', async (event) => {
      if (!window.fetch || sending) return;
      if (!endpoint) return;
      event.preventDefault();
      sending = true;
      setStatus('Sending message…', 'info');
      submitBtn && (submitBtn.disabled = true);
      try {
        const formData = new FormData(form);
        const payload = {
          name: (formData.get('name') || '').toString().trim(),
          email: (formData.get('email') || '').toString().trim(),
          message: (formData.get('message') || '').toString().trim(),
          company: (formData.get('company') || '').toString().trim()
        };
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.error) {
          throw new Error(data.error || 'Unable to send message.');
        }
        setStatus('Thanks! I received your message and will reply soon.', 'success');
        form.reset();
      } catch (err) {
        console.error('Contact form submit failed', err);
        setStatus(err?.message || 'Something went wrong. Please email me directly.', 'error');
      } finally {
        sending = false;
        submitBtn && (submitBtn.disabled = false);
      }
    });
  }
  window.addEventListener('hashchange', openIfHashMatches);
  openIfHashMatches();
})();
