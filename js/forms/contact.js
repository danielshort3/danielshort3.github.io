(() => {
  'use strict';
  const modal = document.getElementById('contact-modal');
  const content = modal?.querySelector('.modal-content');
  const openBtn = document.getElementById('contact-form-toggle');
  const closeBtn = modal?.querySelector('.modal-close');
  const form = document.getElementById('contact-form');
  const statusEl = document.getElementById('contact-status');
  const altContact = document.getElementById('contact-alt');
  const resetBtn = form?.querySelector('[data-contact-reset]');
  const submitBtn = form?.querySelector('[type="submit"]');
  const successPanel = document.getElementById('contact-success');
  const newMessageBtn = successPanel?.querySelector('[data-contact-new]');
  const endpoint = form?.dataset.endpoint || form?.getAttribute('action') || '';
  let prevFocus = null;
  let sending = false;
  const nameInput = form?.querySelector('#contact-name');
  const emailInput = form?.querySelector('#contact-email');
  const messageInput = form?.querySelector('#contact-message');
  const fieldConfigs = [
    {
      input: nameInput,
      indicator: document.getElementById('contact-name-required')
    },
    {
      input: emailInput,
      indicator: document.getElementById('contact-email-required'),
      invalidIndicator: '- Check email'
    },
    {
      input: messageInput,
      indicator: document.getElementById('contact-message-required')
    }
  ];
  fieldConfigs.forEach((config) => {
    if (config.indicator) {
      config.defaultIndicator = config.indicator.textContent.trim() || '- Required';
    }
  });

  const focusables = () => content ? content.querySelectorAll('a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])') : [];
  const trap = (e) => {
    if (e.key !== 'Tab') return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0], last = f[f.length-1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  const setStatus = (message = '', tone = 'info', { focus = false } = {}) => {
    if (!statusEl) return;
    statusEl.textContent = message;
    if (message) {
      statusEl.dataset.tone = tone;
      statusEl.setAttribute('aria-live', tone === 'error' ? 'assertive' : 'polite');
    } else {
      delete statusEl.dataset.tone;
      statusEl.setAttribute('aria-live', 'polite');
    }
    if (altContact) {
      altContact.hidden = !(tone === 'error' && Boolean(message));
    }
    if (message && focus) {
      statusEl.focus({ preventScroll: true });
    }
  };

  const getTrimmedValue = (input) => (input?.value || '').trim();
  const hasValue = (input) => getTrimmedValue(input).length > 0;
  const emailIsValid = (input = emailInput) => {
    if (!input) return false;
    const value = getTrimmedValue(input);
    if (!value) return false;
    if ('validity' in input) return input.validity.valid;
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  };
  const updateSubmitState = () => {
    if (!submitBtn) return;
    submitBtn.disabled = sending || !endpoint;
    submitBtn.classList.toggle('is-busy', sending);
  };

  const showFieldError = (config, { invalid = false } = {}) => {
    if (!config?.input) return;
    const field = config.input.closest('.form-field');
    config.input.setAttribute('aria-invalid', 'true');
    field && field.classList.add('has-error');
    if (config.indicator) {
      const labelText = invalid && config.invalidIndicator ? config.invalidIndicator : config.defaultIndicator || '- Required';
      config.indicator.textContent = labelText;
      config.indicator.hidden = false;
    }
  };
  const clearFieldError = (config) => {
    if (!config?.input) return;
    const field = config.input.closest('.form-field');
    config.input.removeAttribute('aria-invalid');
    field && field.classList.remove('has-error');
    if (config.indicator) {
      config.indicator.textContent = config.defaultIndicator || '- Required';
      config.indicator.hidden = true;
    }
  };
  const validateField = (config) => {
    if (!config?.input) return true;
    const value = getTrimmedValue(config.input);
    if (!value) {
      showFieldError(config);
      return false;
    }
    if (config.input.type === 'email' && !emailIsValid(config.input)) {
      showFieldError(config, { invalid: true });
      return false;
    }
    clearFieldError(config);
    return true;
  };
  const validateForm = () => {
    let firstInvalid = null;
    fieldConfigs.forEach((config) => {
      const valid = validateField(config);
      if (!valid && !firstInvalid) {
        firstInvalid = config.input;
      }
    });
    return firstInvalid;
  };

  const toggleSuccess = (show = false) => {
    if (!form || !successPanel) return;
    form.hidden = show;
    successPanel.hidden = !show;
    if (show) {
      successPanel.focus({ preventScroll: true });
    }
  };
  const prepareForm = () => {
    sending = false;
    form?.setAttribute('aria-busy', 'false');
    toggleSuccess(false);
    setStatus('');
    fieldConfigs.forEach(clearFieldError);
    updateSubmitState();
  };
  const clearInputs = () => {
    if (!form) return;
    form.reset();
    fieldConfigs.forEach(clearFieldError);
  };

  function open(){ if(!modal || !content) return;
    prepareForm();
    prevFocus = document.activeElement;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    content.setAttribute('tabindex','0');
    content.focus({preventScroll:true});
    content.addEventListener('keydown', trap);
  }
  function close(){ if(!modal || !content) return;
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    content.removeEventListener('keydown', trap);
    if (prevFocus && document.contains(prevFocus)) {
      prevFocus.focus();
    } else if (openBtn) {
      openBtn.focus();
    }
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
    form.setAttribute('aria-busy', 'false');
    updateSubmitState();
    const handleInput = () => {
      updateSubmitState();
    };
    form.addEventListener('input', handleInput);
    fieldConfigs.forEach((config) => {
      config.input?.addEventListener('input', () => {
        if (config.input?.getAttribute('aria-invalid') === 'true') {
          validateField(config);
        }
      });
      config.input?.addEventListener('blur', () => validateField(config));
    });
    resetBtn?.addEventListener('click', () => {
      clearInputs();
      setStatus('');
      nameInput?.focus();
    });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!window.fetch || sending || !endpoint) return;
      const firstInvalid = validateForm();
      if (firstInvalid) {
        setStatus('Please fix the highlighted fields before sending.', 'error');
        firstInvalid.focus({ preventScroll: true });
        return;
      }
      sending = true;
      form.setAttribute('aria-busy', 'true');
      setStatus('Sending message…', 'info', { focus: true });
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
        clearInputs();
        setStatus('');
        toggleSuccess(true);
      } catch (err) {
        console.error('Contact form submit failed', err);
        setStatus(err?.message || 'Something went wrong. Please email me directly.', 'error', { focus: true });
      } finally {
        sending = false;
        form.setAttribute('aria-busy', 'false');
        updateSubmitState();
      }
    });
  }
  newMessageBtn?.addEventListener('click', () => {
    toggleSuccess(false);
    clearInputs();
    setStatus('');
    nameInput?.focus();
  });
  window.addEventListener('hashchange', openIfHashMatches);
  openIfHashMatches();
})();
