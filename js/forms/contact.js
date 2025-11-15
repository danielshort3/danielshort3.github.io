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
  const noteEl = form?.querySelector('.contact-form-note');
  const nameInput = form?.querySelector('#contact-name');
  const emailInput = form?.querySelector('#contact-email');
  const messageInput = form?.querySelector('#contact-message');

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

  const getTrimmedValue = (input) => (input?.value || '').trim();
  const hasValue = (input) => getTrimmedValue(input).length > 0;
  const emailIsValid = () => {
    if (!emailInput) return false;
    const value = getTrimmedValue(emailInput);
    if (!value) return false;
    if ('validity' in emailInput) return emailInput.validity.valid;
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  };
  const isFormReady = () => hasValue(nameInput) && hasValue(messageInput) && emailIsValid();
  const hideNote = () => { if (noteEl) noteEl.hidden = true; };
  const showNote = () => { if (noteEl) noteEl.hidden = false; };
  const updateSubmitState = () => {
    if (!submitBtn) return;
    submitBtn.disabled = sending || !isFormReady();
  };

  const resetForm = () => {
    if (form) form.reset();
    sending = false;
    setStatus('');
    hideNote();
    updateSubmitState();
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
    hideNote();
    updateSubmitState();
    const handleInput = () => {
      hideNote();
      updateSubmitState();
    };
    form.addEventListener('input', handleInput);
    form.addEventListener('submit', async (event) => {
      if (!window.fetch || sending) return;
      if (!endpoint) return;
      event.preventDefault();
      sending = true;
      setStatus('Sending message…', 'info');
      updateSubmitState();
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
        updateSubmitState();
        showNote();
      } catch (err) {
        console.error('Contact form submit failed', err);
        setStatus(err?.message || 'Something went wrong. Please email me directly.', 'error');
      } finally {
        sending = false;
        updateSubmitState();
      }
    });
  }
  window.addEventListener('hashchange', openIfHashMatches);
  openIfHashMatches();
})();
