(() => {
  'use strict';

  const CONTACT_CONTEXT_KEY = 'contactOrigin';
  const MAX_CONTEXT_AGE_MS = 15 * 60 * 1000;
  let activeContact = null;

  const query = (root, selector) => root?.querySelector?.(selector) || null;
  const clearStoredContext = () => {
    try { sessionStorage.removeItem(CONTACT_CONTEXT_KEY); } catch (_) {}
  };
  const readStoredContext = () => {
    try {
      const raw = sessionStorage.getItem(CONTACT_CONTEXT_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return null;
      const url = typeof parsed.url === 'string' ? parsed.url.trim() : '';
      const title = typeof parsed.title === 'string' ? parsed.title.trim() : '';
      const audience = typeof parsed.audience === 'string' ? parsed.audience.trim() : '';
      const ts = Number(parsed.ts || 0);
      if (!url && !title) {
        clearStoredContext();
        return null;
      }
      if (ts && Number.isFinite(ts) && Date.now() - ts > MAX_CONTEXT_AGE_MS) {
        clearStoredContext();
        return null;
      }
      return { url, title, audience };
    } catch (_) {
      clearStoredContext();
      return null;
    }
  };
  const getPageContext = () => {
    const stored = readStoredContext();
    if (stored) {
      clearStoredContext();
      return stored;
    }
    const url = window.location?.href?.trim?.() || '';
    const title = (document.title || '').trim();
    const audience = typeof window.getSiteAudience === 'function'
      ? window.getSiteAudience()
      : String(document.body?.dataset?.audience || 'personal').trim();
    return { url, title, audience };
  };
  const appendPageContext = (message = '', context = getPageContext()) => {
    if (!context || (!context.url && !context.title)) return message;
    const audienceLabel = context.audience && context.audience !== 'personal'
      ? `\nAudience: ${context.audience}`
      : '';
    const label = `${context.title ? `Page: ${context.title} - ${context.url}` : `Page: ${context.url}`}${audienceLabel}`;
    if (message.includes(label)) return message;
    return message ? `${message}\n\n${label}` : label;
  };
  const trackContactEvent = (name, params = {}) => {
    try {
      if (typeof window.gaEvent === 'function') window.gaEvent(name, params);
    } catch (_) {}
  };

  const mountContactForm = (root = document) => {
    const modal = query(root, '#contact-modal');
    if (!modal) return null;
    if (activeContact?.modal === modal) return activeContact;
    activeContact?.dispose();

    if (root !== document) {
      document.querySelectorAll('#contact-modal[data-contact-modal-injected="true"]').forEach((node) => {
        if (node !== modal) node.remove();
      });
    }

    const cleanups = [];
    const listen = (target, type, listener, options) => {
      if (!target?.addEventListener) return;
      target.addEventListener(type, listener, options);
      cleanups.push(() => target.removeEventListener(type, listener, options));
    };
    const content = query(modal, '.modal-content');
    const openBtn = query(root, '#contact-form-toggle') || document.getElementById('contact-form-toggle');
    const closeBtn = query(modal, '.modal-close');
    const form = query(modal, '#contact-form');
    const statusEl = query(modal, '#contact-status');
    const altContact = query(modal, '#contact-alt');
    const resetBtn = query(form, '[data-contact-reset]');
    const submitBtn = query(form, '[type="submit"]');
    const successPanel = query(modal, '#contact-success');
    const newMessageBtn = query(successPanel, '[data-contact-new]');
    const endpoint = form?.dataset.endpoint || form?.getAttribute('action') || '';
    const nameInput = query(form, '#contact-name');
    const emailInput = query(form, '#contact-email');
    const messageInput = query(form, '#contact-message');
    const fieldConfigs = [
      { input: nameInput, indicator: query(modal, '#contact-name-required') },
      { input: emailInput, indicator: query(modal, '#contact-email-required'), invalidIndicator: '- Check email' },
      { input: messageInput, indicator: query(modal, '#contact-message-required') }
    ];
    let prevFocus = null;
    let sending = false;
    let submitController = null;
    let hashOpenTimer = 0;
    let disposed = false;
    const modalAccessibility = typeof window.createModalAccessibility === 'function'
      ? window.createModalAccessibility(modal)
      : null;

    fieldConfigs.forEach((config) => {
      if (config.indicator) config.defaultIndicator = config.indicator.textContent.trim() || '- Required';
    });
    const currentPathname = () => {
      try { return String(window.location?.pathname || '').trim(); } catch (_) { return ''; }
    };
    const focusables = () => content
      ? [...content.querySelectorAll('a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])')]
        .filter((node) => !node.disabled && !node.closest('[hidden], [inert], [aria-hidden="true"]'))
      : [];
    const trap = (event) => {
      if (event.key !== 'Tab') return;
      const nodes = focusables();
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
    const setStatus = (message = '', tone = 'info', { focus = false } = {}) => {
      if (!statusEl) return;
      statusEl.textContent = message;
      if (message) statusEl.dataset.tone = tone;
      else delete statusEl.dataset.tone;
      statusEl.setAttribute('aria-live', tone === 'error' ? 'assertive' : 'polite');
      if (altContact) altContact.hidden = !(tone === 'error' && Boolean(message));
      if (message && focus) statusEl.focus({ preventScroll: true });
    };
    const trimmed = (input) => (input?.value || '').trim();
    const emailIsValid = () => {
      const value = trimmed(emailInput);
      if (!value) return false;
      return emailInput && 'validity' in emailInput
        ? emailInput.validity.valid
        : /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
    };
    const updateSubmitState = () => {
      if (!submitBtn) return;
      submitBtn.disabled = sending || !endpoint;
      submitBtn.classList.toggle('is-busy', sending);
    };
    const showFieldError = (config, invalid = false) => {
      if (!config?.input) return;
      config.input.setAttribute('aria-invalid', 'true');
      config.input.closest('.form-field')?.classList.add('has-error');
      if (config.indicator) {
        config.indicator.textContent = invalid && config.invalidIndicator
          ? config.invalidIndicator
          : config.defaultIndicator || '- Required';
        config.indicator.hidden = false;
      }
    };
    const clearFieldError = (config) => {
      if (!config?.input) return;
      config.input.removeAttribute('aria-invalid');
      config.input.closest('.form-field')?.classList.remove('has-error');
      if (config.indicator) {
        config.indicator.textContent = config.defaultIndicator || '- Required';
        config.indicator.hidden = true;
      }
    };
    const validateField = (config) => {
      if (!config?.input) return true;
      if (!trimmed(config.input)) {
        showFieldError(config);
        return false;
      }
      if (config.input.type === 'email' && !emailIsValid()) {
        showFieldError(config, true);
        return false;
      }
      clearFieldError(config);
      return true;
    };
    const validateForm = () => {
      let firstInvalid = null;
      fieldConfigs.forEach((config) => {
        if (!validateField(config) && !firstInvalid) firstInvalid = config.input;
      });
      return firstInvalid;
    };
    const toggleSuccess = (show = false) => {
      if (!form || !successPanel) return;
      form.hidden = show;
      successPanel.hidden = !show;
      form.setAttribute('aria-hidden', show ? 'true' : 'false');
      successPanel.setAttribute('aria-hidden', show ? 'false' : 'true');
      modal.classList.toggle('contact-success', show);
      if (show) {
        const body = query(modal, '.modal-body');
        if (body) body.scrollTop = 0;
        successPanel.focus();
      }
    };
    const clearInputs = () => {
      form?.reset();
      fieldConfigs.forEach(clearFieldError);
    };
    const prepareForm = () => {
      sending = false;
      form?.setAttribute('aria-busy', 'false');
      toggleSuccess(false);
      setStatus('');
      fieldConfigs.forEach(clearFieldError);
      updateSubmitState();
    };
    const syncModalOpenState = () => {
      if (!document.querySelector('.modal.active')) document.body?.classList.remove('modal-open');
    };
    const focusDialog = () => {
      if (!modal.classList.contains('active') || !content || content.contains(document.activeElement)) return;
      content.focus({ preventScroll: true });
    };
    const open = () => {
      if (!content || disposed) return;
      if (modal.classList.contains('active')) {
        focusDialog();
        return;
      }
      prepareForm();
      trackContactEvent('contact_modal_open', { page_path: currentPathname() });
      prevFocus = document.activeElement;
      modalAccessibility?.show();
      modal.classList.add('active');
      document.body?.classList.add('modal-open');
      content.setAttribute('tabindex', '0');
      content.focus({ preventScroll: true });
      modalAccessibility?.isolateBackground();
      content.addEventListener('keydown', trap);
      window.requestAnimationFrame(focusDialog);
    };
    const close = ({ restoreFocus = true } = {}) => {
      if (!content || !modal.classList.contains('active')) return;
      modalAccessibility?.restoreBackground();
      modal.classList.remove('active');
      modalAccessibility?.hide();
      trackContactEvent('contact_modal_close', { page_path: currentPathname() });
      syncModalOpenState();
      content.removeEventListener('keydown', trap);
      if (!restoreFocus) return;
      if (prevFocus && prevFocus !== document.body && document.contains(prevFocus)) {
        prevFocus.focus({ preventScroll: true });
      } else {
        openBtn?.focus({ preventScroll: true });
      }
    };
    const openIfHashMatches = () => {
      if (window.location.hash !== '#contact-modal') return;
      if (hashOpenTimer) window.clearTimeout(hashOpenTimer);
      hashOpenTimer = window.setTimeout(() => {
        hashOpenTimer = 0;
        if (!modal.classList.contains('active')) open();
      }, 120);
    };
    const handleEscape = (event) => {
      if (event.key === 'Escape' && modal.classList.contains('active')) close();
    };
    const handleModalClick = (event) => {
      if (event.target === modal || event.target.closest('[data-contact-close]')) close();
    };
    const handleSubmit = async (event) => {
      event.preventDefault();
      if (!window.fetch || sending || !endpoint) return;
      const firstInvalid = validateForm();
      if (firstInvalid) {
        trackContactEvent('contact_form_validation_error', {
          page_path: currentPathname(),
          field_id: String(firstInvalid.id || '')
        });
        setStatus('');
        firstInvalid.focus({ preventScroll: true });
        return;
      }
      trackContactEvent('contact_form_submit', { page_path: currentPathname() });
      sending = true;
      form.setAttribute('aria-busy', 'true');
      setStatus('Sending message…', 'info', { focus: true });
      updateSubmitState();
      submitController = typeof AbortController === 'function' ? new AbortController() : null;
      try {
        const formData = new FormData(form);
        const pageContext = getPageContext();
        const payload = {
          name: String(formData.get('name') || '').trim(),
          email: String(formData.get('email') || '').trim(),
          message: appendPageContext(String(formData.get('message') || '').trim(), pageContext),
          audience: pageContext.audience || 'personal',
          pageUrl: pageContext.url || '',
          pageTitle: pageContext.title || '',
          company: String(formData.get('company') || '').trim()
        };
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          ...(submitController ? { signal: submitController.signal } : {})
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error) throw new Error(data.error || 'Unable to send message.');
        if (disposed) return;
        clearInputs();
        setStatus('');
        trackContactEvent('contact_form_success', { page_path: currentPathname() });
        toggleSuccess(true);
      } catch (error) {
        if (disposed || submitController?.signal.aborted) return;
        console.error('Contact form submit failed', error);
        trackContactEvent('contact_form_error', {
          page_path: currentPathname(),
          reason: String(error?.message || 'unknown').slice(0, 120)
        });
        setStatus(error?.message || 'Something went wrong. Please email me directly.', 'error', { focus: true });
      } finally {
        submitController = null;
        sending = false;
        if (!disposed) {
          form.setAttribute('aria-busy', 'false');
          updateSubmitState();
        }
      }
    };

    listen(openBtn, 'click', open);
    listen(closeBtn, 'click', close);
    listen(modal, 'click', handleModalClick);
    listen(document, 'keydown', handleEscape);
    listen(window, 'hashchange', openIfHashMatches);
    listen(form, 'input', updateSubmitState);
    fieldConfigs.forEach((config) => {
      listen(config.input, 'input', () => {
        if (config.input?.getAttribute('aria-invalid') === 'true') validateField(config);
      });
      listen(config.input, 'blur', () => validateField(config));
    });
    listen(resetBtn, 'click', () => {
      clearInputs();
      setStatus('');
      nameInput?.focus();
    });
    listen(form, 'submit', handleSubmit);
    listen(newMessageBtn, 'click', () => {
      toggleSuccess(false);
      clearInputs();
      setStatus('');
      nameInput?.focus();
    });
    form?.setAttribute('aria-busy', 'false');
    updateSubmitState();

    const controller = {
      modal,
      open,
      close,
      get sending() { return sending; },
      dispose() {
        if (disposed) return;
        if (hashOpenTimer) window.clearTimeout(hashOpenTimer);
        hashOpenTimer = 0;
        submitController?.abort();
        submitController = null;
        close({ restoreFocus: false });
        disposed = true;
        cleanups.splice(0).reverse().forEach((cleanup) => cleanup());
        if (activeContact === controller) activeContact = null;
        if (window.openContactModal === open) delete window.openContactModal;
        if (window.closeContactModal === close) delete window.closeContactModal;
        window.__contactModalReady = false;
      }
    };
    activeContact = controller;
    window.openContactModal = open;
    window.closeContactModal = close;
    window.__contactModalReady = true;
    openIfHashMatches();
    return controller;
  };

  window.initializeContactModal = (root = document) => mountContactForm(root);

  if (window.SiteRoutes?.register) {
    window.SiteRoutes.register('contact:contact', {
      mount(context) {
        const controller = mountContactForm(context.root);
        if (controller) context.cleanup(() => controller.dispose());
      },
      beforeLeave() {
        return !activeContact?.sending;
      },
      unmount() {
        activeContact?.dispose();
      }
    });
  }

  const initializeDirect = () => {
    const modal = document.getElementById('contact-modal');
    if (modal && activeContact?.modal !== modal) mountContactForm(document);
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDirect, { once: true });
  } else {
    initializeDirect();
  }
})();
